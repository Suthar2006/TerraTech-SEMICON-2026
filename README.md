# TerraTech-SEMICON-2026


Drift-Sense: Navigation-Error Recovery for Semiconductor Visual Inspection

=>Team:TerraTech  
=>College:Karpagam College of Engineering  
=>Hackathon:SEMICON India Hackathon 2026

---

1. Problem Statement

Drift-Sense: Navigation-Error Recovery

Semiconductor wafer inspection systems work with detailed and repetitive layouts. Navigation and localization errors can occur when the system loses the correct position of a small reference region inside a much larger search image.

The challenge is to reliably localize the reference region while handling repetitive structures, scale variation, rotation, image degradation, and ambiguous false matches.

---

2. Proposed Solution

Drift-Sense is a multi-stage visual localization and recovery pipeline designed for repetitive semiconductor layouts.

The system receives:

- A small reference image
- A large search image

It generates localization candidates, verifies them using complementary visual features, suppresses periodic false matches, and applies targeted recovery and consensus checks for difficult cases.

Core Pipeline

Reference Image + Search Image
        ↓
Candidate Generation
        ↓
Multi-Scale / Rotation-Aware Matching
        ↓
Grayscale Verification
        ↓
Edge Verification
        ↓
Gradient Verification
        ↓
Projection Verification
        ↓
Candidate Suppression
        ↓
Ambiguity Detection
        ↓
Targeted Recovery
        ↓
Recovery / Consensus
        ↓
Local Refinement
        ↓
Final Localization

---

3. Key Features

- Multi-scale visual localization
- Rotation-aware matching
- Grayscale/template verification
- Edge-based verification
- Gradient-based verification
- Projection/structural verification
- Periodic false-match suppression
- Ambiguity detection
- Targeted recovery
- Recovery consensus
- Local refinement
- Failure diagnostics
- CPU-parallel processing

---

4. Technology Stack

- Python
- OpenCV
- NumPy
- CSV-based result analysis
- CPU-parallel processing

The CPU-parallel implementation distributes independent image-pair processing without changing the core localization logic.

---

5. Dataset

The final system was evaluated on:

=>1,000 image pairs<=

Each pair contains:

```text
pair_xxx/
├── reference.png
├── search.png
└── label.txt
