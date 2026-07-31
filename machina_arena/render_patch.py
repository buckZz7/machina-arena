"""Patch SAPIEN/ManiSkill to enable lavapipe software rendering on CPU.

Import this BEFORE importing gymnasium or mani_skill.
"""
import sapien
import mani_skill.render.utils as ru

def patched_can_render(device):
    return device is not None
ru.can_render = patched_can_render

_original_RenderSystem = sapien.render.RenderSystem
def patched_RenderSystem(device=None):
    return _original_RenderSystem()
sapien.render.RenderSystem = patched_RenderSystem

import mani_skill.envs.sapien_env as se
se.render_utils.can_render = patched_can_render
