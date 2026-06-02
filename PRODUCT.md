# Product

## Register

product

## Users

ML engineers, data scientists, and medical AI researchers running a GPU cluster on-prem or in a private cloud. They open this tool mid-workflow — annotating datasets in Label Studio, queuing training jobs, monitoring a Ray cluster — not to browse, but to execute. Context: seated, focused, usually running a model alongside the browser.

## Product Purpose

MedImage is an internal MLOps platform for medical imaging AI. It connects Label Studio annotation projects to object storage, manages model training jobs (classification, detection, segmentation, VLM fine-tuning, edge export) on a Ray cluster, and surfaces cluster and job state in real time. Success looks like: annotator clicks sync, training job queues, engineer monitors progress, model ships — no terminal required.

## Brand Personality

Precise. Clinical. Capable.

- **Precise**: every label, number, and affordance is exactly what it says. No rounding of intent.
- **Clinical**: the aesthetic of a instrument display panel — clean, low-noise, confident neutrals. Not cold; competent.
- **Capable**: the tool doesn't hide power. Complex configs are exposed, not buried behind "smart" defaults.

## Anti-references

- Generic SaaS dashboards: blue gradients on every card, identical stat-card grids, purple-to-blue hero accents.
- AI startup clichés: glassmorphism, gradient text, glowing cards as decoration.
- Consumer health apps: pastels, excessive rounding, friendly-first design that undercuts the technical register.

## Design Principles

1. **Instrument-panel clarity**: information density is a feature, not a problem. Data should be readable at a glance without decoration.
2. **State is signal**: color, weight, and motion exist to communicate state (running, error, syncing, idle), not to decorate.
3. **Tool disappears into task**: the interface does not ask to be noticed. Users stay focused on their work, not on the UI.
4. **Earned familiarity**: components follow industry-standard affordances (Linear, Vercel-style dashboards). No invented patterns.
5. **Density without clutter**: show what the user needs per screen, keep secondary info accessible but not intrusive.

## Accessibility & Inclusion

- WCAG AA minimum. All text ≥ 4.5:1 contrast.
- Semantic states for color-blind users: never use color alone to convey status; pair with icon or label.
- Reduced motion respected via `prefers-reduced-motion`.
- Keyboard navigation for all interactive elements.
