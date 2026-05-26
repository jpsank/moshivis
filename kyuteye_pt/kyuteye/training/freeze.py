"""Parameter-group freeze utilities for combined MoshiVis+RAG fine-tunes.

The recommended fine-tune workflow starts by freezing the 7B Moshi
backbone (Helium) and PaliGemma vision encoder, then training only the
newly-added parameters: ARC encoder, conditioner registry (LUT for
``first_speaker``, etc.), learnt padding, the bridge projector. This
preserves MoshiVis's vision capability while teaching the model to
consume the streaming-sum signal.

Once the new modules plateau on the small parameter set, optionally
unfreeze the LM backbone (or attach LoRA adapters to it) for a second
fine-tune phase at smaller learning rate.

The recipes below name the partition explicitly so an unfamiliar reader
can audit what is and isn't trainable at a glance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import torch
from torch import nn

logger = logging.getLogger(__name__)


# A "recipe" is a function that takes the MoshiVis model + ImageProjection
# + the MoshiVisGen wrapper and returns the iterable of parameters that
# should be trainable. Everything else gets ``requires_grad_(False)``.
FreezeRecipe = Callable[
    [nn.Module, nn.Module, nn.Module], Iterable[nn.Parameter]
]


def _trainable_params_for_adapters(
    moshi_vis: nn.Module,
    image_proj: nn.Module,
    moshi_vis_gen: nn.Module,
) -> List[nn.Parameter]:
    """Partial-freeze recipe: train RAG conditioners + ARC encoder only.

    Trainable:

    * ``MoshiVis.condition_provider.*`` -- LUT conditioner, ARC encoder
      (if configured as ``type: arc``), bridge projector, learnt_padding.
    * ``MoshiVis.fuser.*`` -- has no trainable params today but kept in
      the list for forward-compat in case future fusers grow weights.
    * ``ImageProjection.proj_xa`` -- the vision projection to LM dim.
      MoshiVis already trained this; we *don't* re-train it here.
    * Cross-attention adapter weights inside the transformer
      (``layers.0.cross_attention.*`` since cross-attn is shared and lives
      on layer 0). MoshiVis trained these too -- skip by default.

    The 7B Helium backbone, PaliGemma vision encoder, depformer, and
    Mimi codec stay frozen.
    """
    trainable: List[nn.Parameter] = []
    cp = getattr(moshi_vis, "condition_provider", None)
    if cp is not None:
        trainable.extend(p for p in cp.parameters())
    fuser = getattr(moshi_vis, "fuser", None)
    if fuser is not None:
        trainable.extend(p for p in fuser.parameters())
    return trainable


def _trainable_params_with_xa_adapter(
    moshi_vis: nn.Module,
    image_proj: nn.Module,
    moshi_vis_gen: nn.Module,
) -> List[nn.Parameter]:
    """Recipe: adapters + the shared cross-attention adapter on layer 0.

    Same as :func:`_trainable_params_for_adapters` plus the trainable
    cross-attention module on transformer layer 0 (which is the shared
    adapter all other layers reference). Useful if the RAG fine-tune
    should also bend the visual XA path slightly.
    """
    base = _trainable_params_for_adapters(moshi_vis, image_proj, moshi_vis_gen)
    layers = moshi_vis.llm.transformer.layers
    layer0 = layers[0] if len(layers) > 0 else None
    if layer0 is not None and hasattr(layer0, "cross_attention"):
        base.extend(layer0.cross_attention.parameters())
    return base


def _trainable_params_full_lm(
    moshi_vis: nn.Module,
    image_proj: nn.Module,
    moshi_vis_gen: nn.Module,
) -> List[nn.Parameter]:
    """Recipe: everything except the frozen PaliGemma vision encoder.

    The most invasive recipe. Use only after the adapter-only phase
    has converged; pair with a small LR (~1e-5) and gradient clipping.
    """
    trainable: List[nn.Parameter] = []
    trainable.extend(moshi_vis.parameters())
    # Vision encoder stays frozen even here -- PaliGemma2 is huge and
    # MoshiVis was trained with it frozen. The projection on top of it
    # (proj_xa) gets included if it's a child of ImageProjection.
    for name, mod in image_proj.named_children():
        if name == "encoder":
            continue  # Skip the PaliGemma vision encoder
        trainable.extend(mod.parameters())
    return trainable


freeze_recipes: Dict[str, FreezeRecipe] = {
    "adapters_only": _trainable_params_for_adapters,
    "adapters_plus_xa": _trainable_params_with_xa_adapter,
    "full_lm": _trainable_params_full_lm,
}


@dataclass
class FreezeReport:
    """Summary of which parameters ended up trainable / frozen.

    Surface this at training startup so the user can sanity-check the
    recipe selected the right partition before kicking off a multi-hour
    fine-tune.
    """

    recipe: str
    trainable_count: int
    frozen_count: int
    trainable_param_groups: List[str]
    frozen_param_groups: List[str]

    def pretty(self) -> str:
        return (
            f"[freeze] recipe={self.recipe!r}\n"
            f"  trainable: {self.trainable_count:,} params across "
            f"{len(self.trainable_param_groups)} top-level groups\n"
            f"    {', '.join(self.trainable_param_groups[:8])}"
            + ("..." if len(self.trainable_param_groups) > 8 else "")
            + f"\n  frozen: {self.frozen_count:,} params across "
            f"{len(self.frozen_param_groups)} top-level groups\n"
            f"    {', '.join(self.frozen_param_groups[:8])}"
            + ("..." if len(self.frozen_param_groups) > 8 else "")
        )


def apply_freeze_recipe(
    recipe_name: str,
    moshi_vis: nn.Module,
    image_proj: nn.Module,
    moshi_vis_gen: nn.Module,
) -> FreezeReport:
    """Apply ``recipe_name`` to the model trio. Returns a :class:`FreezeReport`.

    1. Calls ``requires_grad_(False)`` on every parameter in all three
       modules.
    2. Calls ``requires_grad_(True)`` on the recipe's trainable set.
    3. Returns a report enumerating which top-level submodules are
       trainable vs frozen, so the caller can sanity-check.

    :raises KeyError: if ``recipe_name`` is not in :data:`freeze_recipes`.
    """
    if recipe_name not in freeze_recipes:
        raise KeyError(
            f"Unknown freeze recipe {recipe_name!r}; "
            f"available: {list(freeze_recipes.keys())}"
        )
    # Freeze everything.
    for mod in (moshi_vis, image_proj, moshi_vis_gen):
        for p in mod.parameters():
            p.requires_grad_(False)
    # Unfreeze the recipe's selection.
    trainable_params = list(freeze_recipes[recipe_name](moshi_vis, image_proj, moshi_vis_gen))
    for p in trainable_params:
        p.requires_grad_(True)
    return summarize_freeze(recipe_name, moshi_vis, image_proj, moshi_vis_gen)


def summarize_freeze(
    recipe_name: str,
    moshi_vis: nn.Module,
    image_proj: nn.Module,
    moshi_vis_gen: nn.Module,
) -> FreezeReport:
    """Build a :class:`FreezeReport` from the current ``requires_grad`` state.

    Dedups parameters by Python ``id()`` because ``MoshiVisGen`` wraps
    ``MoshiVis`` -- iterating both modules' children would otherwise count
    the same parameters twice and overstate the trainable size.
    """
    seen_ids: set[int] = set()
    trainable_count = 0
    frozen_count = 0
    trainable_groups: List[str] = []
    frozen_groups: List[str] = []
    for mod, label in (
        (moshi_vis, "moshi_vis"),
        (image_proj, "image_proj"),
        (moshi_vis_gen, "moshi_vis_gen"),
    ):
        for name, child in mod.named_children():
            t = 0
            f = 0
            for p in child.parameters():
                if id(p) in seen_ids:
                    continue
                seen_ids.add(id(p))
                if p.requires_grad:
                    t += p.numel()
                else:
                    f += p.numel()
            qualified = f"{label}.{name}"
            if t > 0:
                trainable_groups.append(qualified)
                trainable_count += t
            if f > 0 and t == 0:
                frozen_groups.append(qualified)
                frozen_count += f
            elif f > 0:
                frozen_count += f
    return FreezeReport(
        recipe=recipe_name,
        trainable_count=trainable_count,
        frozen_count=frozen_count,
        trainable_param_groups=trainable_groups,
        frozen_param_groups=frozen_groups,
    )
