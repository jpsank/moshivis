"""Batched multi-user inference for the MoshiVis PyTorch backend.

A drop-in alternative to :class:`kyuteye.server.ServerState` that holds the
model at a configurable batch size and serves up to ``batch_size`` concurrent
WebSocket sessions in parallel through a single batched step loop.

Architecture mirrors moshi-rag's ``inference_utils/channel.py`` +
``inference_utils/batch_runner.py``:

* :class:`BatchedServerState` owns the GPU resources (Mimi, MoshiVisGen,
  image encoder) and a pool of N slots.
* :class:`BatchedChannel` is one WebSocket session. It reserves a slot at
  connect time, owns per-slot state (image cross-attention KV, Omni
  TextStreamMonitor / RAGManager / ContextInjector, opus reader/writer),
  and exchanges PCM frames with the central step loop through asyncio
  queues.
* The step loop (:meth:`BatchedServerState._step_loop`) runs once per
  frame on the full batch. For each slot it gathers either the next
  user-provided PCM frame (active) or silence (idle), assembles an
  ``exec_mask`` over the slots, and pushes the batch through Mimi encode,
  MoshiVisGen.step, Mimi decode in one shot. Outputs are routed back to
  the originating slot's output queue.

What is supported now
---------------------
* True parallel inference for up to ``batch_size`` concurrent sessions.
* Per-slot silence handling via ``exec_mask`` -- silent users don't
  consume their streaming-sum queue and don't corrupt the attention KV
  cache for their slot (idle K/V values are preserved at the write
  position; see ``kyuteye/modules/attention.py:KVCache.complete``).
* Per-slot Omni RAG: every channel has its own retriever / monitor /
  context injector, so retrieval and tool calls fire independently.

What is **not** supported and is documented as a limitation
-----------------------------------------------------------
Dynamic mid-batch user join/leave needs per-slot ``end_offset`` in the
attention KV cache and per-slot ``offset`` in :class:`MoshiVisGen` -- so
that releasing slot ``i`` can clear *only* its KV history without
disturbing the other active slots. The current implementation shares a
single ``current_end`` / ``offset`` across the batch. Practical impact:

* **Works**: a fixed set of N sessions start within a small time window
  and run to completion; users may pause / resume silence freely.
* **Does not work**: starting a new session while others are mid-stream
  -- the new user would inherit the existing slot's KV cache state.

The :meth:`BatchedServerState.acquire_slot` flow therefore refuses new
connections once any session has produced its first output, until all
sessions in the current batch have disconnected. A future refactor to
per-slot ``end_offset`` (matching upstream moshi 0.2.13's ``KVCache`` and
MoshiRAG's ``LMGen`` state) lifts this restriction; see the module
docstring at ``kyuteye/modules/attention.py``.

How to use
----------
.. code-block:: bash

    server --kyuteye-config-path=configs/moshika-vis.yaml --batch-size=4

When ``--batch-size > 1``, the server constructs a ``BatchedServerState``
instead of the default single-stream ``ServerState``. The HTTP route and
WebSocket protocol are unchanged; clients don't know the difference.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

import aiohttp
import numpy as np
import sentencepiece
import sphn
import torch
from aiohttp import web
from kyuteye.config.enums import ImageEncoder
from kyuteye.modules.image_transforms import get_minimal_transforms
from kyuteye.omni import (
    ContextInjector,
    OmniRAGManager,
    TextStreamMonitor,
    default_registry,
    get_retriever,
)
from kyuteye.omni.arc_encoder_client import encode_reference_async, get_arc_encoder_url
from torchvision.io import ImageReadMode, decode_image

if TYPE_CHECKING:
    from kyuteye.models.image_projection import ImageProjection
    from kyuteye.models.moshivis import MoshiVisGen
    from moshi.models import MimiModel

logger = logging.getLogger(__name__)


@dataclass
class _SlotState:
    """Per-session state held inside the batched server. One instance per
    active slot in the pool."""

    slot_idx: int
    ws: web.WebSocketResponse
    image_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    opus_reader: Optional[Any] = None
    opus_writer: Optional[Any] = None
    pcm_buffer: Optional[np.ndarray] = None
    monitor: Optional[TextStreamMonitor] = None
    context_injector: Optional[ContextInjector] = None
    rag_manager: Optional[OmniRAGManager] = None
    has_produced_output: bool = False
    closed: bool = False
    output_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)


class BatchedServerState:
    """Batched, multi-session variant of ``ServerState``.

    See module docstring for the supported-vs-unsupported feature matrix.
    All Omni options are accepted with the same names as ``ServerState`` so
    the server entry point can swap between the two based on
    ``--batch-size``.
    """

    def __init__(
        self,
        mimi: "MimiModel",
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        moshi_vis: "MoshiVisGen",
        image_encoder_model: "ImageProjection",
        device: str | torch.device,
        batch_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        max_msg_size: int = 0,
        image_size: int = 448,
        xa_start: int = 0,
        omni_enabled: bool = False,
        omni_rag_trigger: str = "<ret>",
        omni_rag_timeout: float = 1.5,
        omni_rag_max_tokens: int = 512,
        omni_rag_wait_steps: int = 0,
        omni_xa_injection: bool = True,
        omni_tool_start: str = "[TOOL:",
        omni_tool_end: str = "]",
        omni_arc_encoder_url: Optional[str] = None,
        omni_injection_mode: Literal["xa", "streaming_sum", "off"] = "xa",
    ) -> None:
        assert batch_size >= 1, "batch_size must be >= 1"
        self.batch_size = batch_size

        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.moshi_vis = moshi_vis
        self.image_encoder_model = image_encoder_model
        self.device = device
        self.dtype = dtype
        self.max_msg_size = max_msg_size
        self.image_size = image_size
        self.xa_start = xa_start
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        # Omni config -- forwarded to each per-slot channel at acquire time.
        self.omni_enabled = omni_enabled
        self.omni_rag_trigger = omni_rag_trigger
        self.omni_rag_timeout = omni_rag_timeout
        self.omni_rag_max_tokens = omni_rag_max_tokens
        self.omni_rag_wait_steps = omni_rag_wait_steps
        self.omni_xa_injection = omni_xa_injection
        self.omni_tool_start = omni_tool_start
        self.omni_tool_end = omni_tool_end
        self.omni_arc_encoder_url = omni_arc_encoder_url
        self.omni_injection_mode = omni_injection_mode

        # Slot pool. ``None`` = free, _SlotState = occupied.
        self.slots: list[Optional[_SlotState]] = [None] * batch_size
        self._slots_lock = asyncio.Lock()
        self._any_output_seen = False  # gates new acquires (see docstring)

        # The model is permanently in streaming mode at the batched size.
        self.mimi.streaming_forever(batch_size)
        self.moshi_vis.streaming_forever(batch_size)

        # Step loop handle; created when the server starts serving.
        self._step_loop_task: Optional[asyncio.Task] = None

    # ---------------------------------------------------------------- slot pool
    async def acquire_slot(self, ws: web.WebSocketResponse) -> Optional[_SlotState]:
        """Reserve a free slot for ``ws``.

        Returns ``None`` if the pool is full **or** if at least one slot has
        already produced output this batch (see docstring -- mid-batch joins
        aren't supported without per-slot reset). Caller should send a 503.
        """
        async with self._slots_lock:
            if self._any_output_seen:
                logger.warning(
                    "[Batched] refusing new connection: batch is mid-flight "
                    "and per-slot reset isn't supported. Wait for all current "
                    "sessions to disconnect."
                )
                return None
            for i in range(self.batch_size):
                if self.slots[i] is None:
                    slot = _SlotState(slot_idx=i, ws=ws)
                    self.slots[i] = slot
                    logger.info("[Batched] acquired slot %d (%d/%d)", i, self._active_count(), self.batch_size)
                    return slot
            return None

    async def release_slot(self, slot: _SlotState) -> None:
        """Release ``slot``. If this is the last active session, reset the
        model so the next batch starts clean."""
        async with self._slots_lock:
            if self.slots[slot.slot_idx] is slot:
                self.slots[slot.slot_idx] = None
            slot.closed = True
            logger.info(
                "[Batched] released slot %d (%d/%d remaining)",
                slot.slot_idx,
                self._active_count(),
                self.batch_size,
            )
            if self._active_count() == 0 and self._any_output_seen:
                logger.info("[Batched] last session disconnected; resetting model state")
                self.mimi.reset_streaming()
                self.moshi_vis.reset_streaming()
                self.moshi_vis.prime()
                self._any_output_seen = False

    def _active_count(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    # ---------------------------------------------------------------- model setup
    def warmup(self) -> None:
        """Warm up the batched model with zero PCM + zero image. One round trip."""
        logger.info("[Batched] warming up model at batch_size=%d", self.batch_size)
        chunk = torch.zeros(
            self.batch_size, 1, self.frame_size, dtype=torch.float32, device=self.device
        )
        ca_src = self.image_encoder_model(
            torch.zeros(self.batch_size, 3, 224, 224, device=self.device)
        )["cross_attention_src"]
        # Set full exec_mask so every slot exercises every path.
        full_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.mimi.set_exec_mask(full_mask)
        self.moshi_vis.set_exec_mask(full_mask)
        for _ in range(4):
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens, _ = self.moshi_vis.step(codes[:, :, c : c + 1], ca_src=ca_src)
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.mimi.reset_streaming()
        self.moshi_vis.reset_streaming()

    # ---------------------------------------------------------------- step loop
    async def _step_loop(self) -> None:
        """Forever loop: every frame_size samples, batch one step over all slots."""
        logger.info("[Batched] step loop started")
        try:
            while True:
                await asyncio.sleep(0.001)
                if self._active_count() == 0:
                    await asyncio.sleep(0.005)
                    continue
                await self._run_one_step()
        except asyncio.CancelledError:
            logger.info("[Batched] step loop cancelled")
            raise

    async def _run_one_step(self) -> None:
        """Read one frame per active slot, batch through model, deliver outputs."""
        # Gather per-slot PCM frame (or silence) + exec_mask.
        slots_snapshot: list[Optional[_SlotState]] = list(self.slots)
        pcm_batch = torch.zeros(
            self.batch_size, 1, self.frame_size, dtype=torch.float32, device=self.device
        )
        exec_mask_cpu = torch.zeros(self.batch_size, dtype=torch.bool)
        active_slots: list[_SlotState] = []
        for i, slot in enumerate(slots_snapshot):
            if slot is None or slot.closed or slot.opus_reader is None:
                continue
            try:
                pcm = slot.opus_reader.read_pcm()
            except Exception:  # pragma: no cover - defensive
                continue
            if pcm.shape[-1] == 0:
                continue
            slot.pcm_buffer = pcm if slot.pcm_buffer is None else np.concatenate((slot.pcm_buffer, pcm))
            if slot.pcm_buffer.shape[-1] < self.frame_size:
                continue
            chunk = slot.pcm_buffer[: self.frame_size]
            slot.pcm_buffer = slot.pcm_buffer[self.frame_size :]
            pcm_batch[i, 0] = torch.from_numpy(chunk).to(device=self.device)
            exec_mask_cpu[i] = True
            active_slots.append(slot)

        if not active_slots:
            return

        # Stack per-slot image KV into a batched ca_src. Slots without an
        # image (shouldn't normally happen post-handshake) get zero KV.
        ca_src = self._build_batched_ca_src(slots_snapshot)

        exec_mask_dev = exec_mask_cpu.to(self.device)
        self.mimi.set_exec_mask(exec_mask_dev)
        self.moshi_vis.set_exec_mask(exec_mask_dev)

        be = time.time()
        codes = self.mimi.encode(pcm_batch)
        assert codes.shape[-1] == 1, codes.shape
        for c in range(codes.shape[-1]):
            tokens, gate_weight = self.moshi_vis.step(
                codes[:, :, c : c + 1],
                ca_src=ca_src,
            )
            if tokens is None:
                continue
            assert tokens.shape[1] == self.moshi_vis.num_audio_codebooks_out + 1
            main_pcm = self.mimi.decode(tokens[:, 1:]).cpu()

            for slot in active_slots:
                self._any_output_seen = True
                i = slot.slot_idx
                # PCM out
                slot.opus_writer.append_pcm(main_pcm[i, 0].numpy())
                # Text token + Omni events
                text_token = int(tokens[i, 0, 0].item())
                if text_token in (0, 3):
                    continue
                piece = self.text_tokenizer.id_to_piece(text_token).replace("▁", " ")  # type: ignore[arg-type]
                color = round(max(min((gate_weight - 0.005) / 0.016, 1.0), 0.0) * 10)
                await self._deliver_text(slot, piece, color, text_token)
        elapsed_ms = 1000 * (time.time() - be)
        if elapsed_ms > 77:
            logger.warning("[Batched] step (%d active) took %.1fms", len(active_slots), elapsed_ms)

    def _build_batched_ca_src(
        self, slots: list[Optional[_SlotState]]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Stack per-slot precomputed image KV into a batched ``ca_src``.

        Slots with no image yet (or no occupant) get zero KVs the same shape
        as the first present image's KV. All images at the same resolution
        produce identically-shaped KV, so this is a clean stack.
        """
        if self.moshi_vis.get_streaming_attribute("offset", 0) < self.xa_start:
            return None
        present = [s for s in slots if s is not None and s.image_kv is not None]
        if not present:
            return None
        ref_k, ref_v = present[0].image_kv  # type: ignore[misc]
        batched_k = torch.zeros(
            self.batch_size, *ref_k.shape[1:], device=ref_k.device, dtype=ref_k.dtype
        )
        batched_v = torch.zeros(
            self.batch_size, *ref_v.shape[1:], device=ref_v.device, dtype=ref_v.dtype
        )
        for i, slot in enumerate(slots):
            if slot is None or slot.image_kv is None:
                continue
            k, v = slot.image_kv
            batched_k[i] = k[0]
            batched_v[i] = v[0]
        return batched_k, batched_v

    async def _deliver_text(
        self, slot: _SlotState, piece: str, color: int, text_token: int
    ) -> None:
        """Push one text fragment to a slot's WebSocket, after Omni monitor processing."""
        if self.omni_enabled and slot.monitor is not None:
            emit, events = slot.monitor.consume(piece, token_id=text_token)
            for event in events:
                if event.kind == "rag" and slot.rag_manager is not None:
                    await self._send_omni_text(slot, " [RET] ", marker=10)
                    await slot.rag_manager.trigger(
                        wait_steps=self.omni_rag_wait_steps,
                        handle_reference_fn=lambda ref, s=slot: self._on_reference_text(s, ref),
                        context_provider=lambda s=slot: f"moshi: {s.monitor.transcript}\n",
                    )
                elif event.kind == "tool":
                    asyncio.create_task(self._dispatch_tool(slot, event))
            piece = emit
            if not piece:
                return
        msg = b"\x07" + color.to_bytes(1, "big") + piece.encode("utf-8")
        await slot.ws.send_bytes(msg)

    async def _send_omni_text(self, slot: _SlotState, payload: str, marker: int = 0) -> None:
        msg = b"\x07" + marker.to_bytes(1, "big") + payload.encode("utf-8")
        await slot.ws.send_bytes(msg)

    async def _on_reference_text(self, slot: _SlotState, reference: str) -> None:
        if not reference:
            await self._send_omni_text(slot, " [RET_FAILED] ", marker=10)
            return
        await self._send_omni_text(slot, f" [REF: {reference}] ", marker=10)
        if self.omni_injection_mode == "streaming_sum":
            url = self.omni_arc_encoder_url or get_arc_encoder_url()
            if not url:
                logger.warning("[Batched] streaming_sum mode but no ARC encoder URL")
                return
            try:
                tensor = await encode_reference_async(reference, encoder_url=url)
            except Exception as e:
                logger.error("[Batched] ARC encoder failed: %s", e)
                return
            self.moshi_vis.update_streaming_sum_tensor(tensor, slot_idx=slot.slot_idx)
        elif self.omni_injection_mode == "xa" and slot.context_injector is not None:
            slot.context_injector.add_text(reference, role="reference")

    async def _dispatch_tool(self, slot: _SlotState, event: Any) -> None:
        if event.tool is None:
            await self._send_omni_text(slot, f" [TOOL_ERROR: bad parse {event.raw!r}] ", marker=10)
            return
        result = await default_registry.dispatch(event.tool)
        await self._send_omni_text(slot, f" [TOOL:{event.tool.name} -> {result}] ", marker=10)
        if slot.context_injector is not None:
            slot.context_injector.add_text(f"{event.tool.name} -> {result}", role="tool")

    # ---------------------------------------------------------------- WS handler
    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=self.max_msg_size)
        await ws.prepare(request)

        slot = await self.acquire_slot(ws)
        if slot is None:
            await ws.close()
            return ws

        # Start the step loop on first connection.
        if self._step_loop_task is None or self._step_loop_task.done():
            self._step_loop_task = asyncio.create_task(self._step_loop())

        try:
            async with BatchedChannel(self, slot) as channel:
                await channel.run()
        finally:
            await self.release_slot(slot)
        return ws


class BatchedChannel:
    """One WebSocket session bound to a slot in a :class:`BatchedServerState`."""

    def __init__(self, server: BatchedServerState, slot: _SlotState) -> None:
        self.server = server
        self.slot = slot
        self._stack: Optional[contextlib.AsyncExitStack] = None

    async def __aenter__(self) -> "BatchedChannel":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()

        s = self.slot
        s.opus_writer = sphn.OpusStreamWriter(self.server.mimi.sample_rate)  # type: ignore
        s.opus_reader = sphn.OpusStreamReader(self.server.mimi.sample_rate)  # type: ignore

        # Omni state, one per channel.
        s.monitor = TextStreamMonitor(
            rag_trigger=self.server.omni_rag_trigger,
            rag_token_id=getattr(self.server.moshi_vis.lm_model, "rag_token_id", None),
            tool_start=self.server.omni_tool_start,
            tool_end=self.server.omni_tool_end,
        )
        s.context_injector = ContextInjector(
            moshi_vis=self.server.moshi_vis,
            tokenizer=self.server.text_tokenizer,
            device=self.server.device,
            dtype=self.server.dtype,
            enabled=(
                self.server.omni_enabled
                and self.server.omni_xa_injection
                and self.server.omni_injection_mode == "xa"
            ),
        )
        retriever = get_retriever() if self.server.omni_enabled else None
        if retriever is not None:
            s.rag_manager = OmniRAGManager(
                retriever,
                rag_timeout=self.server.omni_rag_timeout,
                max_tokens=self.server.omni_rag_max_tokens,
            )
            await self._stack.enter_async_context(s.rag_manager)

        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        assert self._stack is not None
        try:
            await self._stack.__aexit__(exc_type, exc_value, tb)
        finally:
            self._stack = None

    async def run(self) -> None:
        s = self.slot
        # Read the initial image; gives us the per-slot ca_src.
        await self._extract_image()
        # Handshake.
        await s.ws.send_bytes(b"\x00")
        # Audio recv + opus send loops.
        await asyncio.gather(self._recv_loop(), self._send_loop())

    async def _extract_image(self) -> None:
        s = self.slot
        first_message = await s.ws.receive()
        data = first_message.data
        kind = data[0]
        if kind != 8:
            raise RuntimeError(f"unknown message kind {kind}")
        payload = data[1:]
        image_tensor = decode_image(
            torch.frombuffer(payload, dtype=torch.uint8), mode=ImageReadMode.RGB
        )
        image_tensor = get_minimal_transforms(self.server.image_size)(image_tensor)
        image_tensor = self.server.image_encoder_model.to_tensor_and_normalize(image_tensor)
        if self.server.image_encoder_model.encoder_type == ImageEncoder.PIXTRAL:
            image_tensor = [image_tensor.to(self.server.device)]
        else:
            image_tensor = image_tensor[None, ...].to(self.server.device)
        k, v = self.server.moshi_vis.precompte_ca_kv(
            self.server.image_encoder_model(image_tensor)["cross_attention_src"]
        )
        s.image_kv = (k.to(self.server.dtype), v.to(self.server.dtype))
        if s.context_injector is not None:
            s.context_injector.set_image_kv(s.image_kv)

    async def _recv_loop(self) -> None:
        s = self.slot
        async for message in s.ws:
            if message.type == aiohttp.WSMsgType.ERROR:
                break
            if message.type == aiohttp.WSMsgType.CLOSED:
                break
            if message.type != aiohttp.WSMsgType.BINARY:
                continue
            data = message.data
            if not isinstance(data, bytes) or len(data) == 0:
                continue
            kind = data[0]
            if kind == 1:
                s.opus_reader.append_bytes(data[1:])

    async def _send_loop(self) -> None:
        s = self.slot
        while not s.closed:
            await asyncio.sleep(0.001)
            msg = s.opus_writer.read_bytes()
            if len(msg) > 0:
                await s.ws.send_bytes(b"\x01" + msg)
