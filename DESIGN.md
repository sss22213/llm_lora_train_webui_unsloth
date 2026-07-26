# LoRA Forge design system

LoRA Forge follows the Linear-style dark product language used by the reference
Heretic WebUI. The interface is a technical workspace, not a marketing page.

## Tokens

- Canvas: `#010102`
- Surface 1–4: `#0f1011`, `#141516`, `#18191a`, `#191a1b`
- Hairlines: `#23252a`, strong `#34343a`
- Primary text: `#f7f8f8`; muted `#d0d6e0`; subtle `#8a8f98`
- Single accent: `#5e6ad2`; hover `#828fff`; focus `#5e69d1`
- Success status only: `#27a644`
- Display fallback: `SF Pro Display, Inter, system-ui`
- Body fallback: `Inter, SF Pro Text, system-ui`
- Mono fallback: `ui-monospace, SFMono-Regular, Menlo`

## Rules

- Use surface lift and 1px hairlines for depth; do not add drop shadows or gradients.
- Reserve lavender for the brand mark, primary action, focus, and selected state.
- Cards use 12px corners; product/console panels may use 16px; controls use 8px.
- Keep controls compact but never below a 40px mouse/touch target.
- Desktop uses a fixed sidebar; below 820px it becomes an off-canvas menu.
- Display headings use weight 600 and negative tracking. Body remains weight 400.
- Do not introduce a second decorative accent color. Red is only destructive/error,
  and green is only completed/healthy status.
- Logs and identifiers use the mono stack and must remain copyable/selectable.

