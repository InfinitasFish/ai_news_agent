### Daily Research: 2026-02-19
                    ### Query: "deep+computer+vision"
                    ### Found *5* relevant papers.
                    #===================================================#


                
                
Paper: QwaveMPS: An efficient open-source Python package for simulating non-Markovian waveguide-QED using matrix product states
Authors: Sofia Arranz Regidor, Matthew Kozma, Stephen Hughes
Categories: quant-ph
Published: 2026-02-17 18:58 UTC
Source: Arxiv
            
Relevance Score: 0.678/1.00
            
Llm analysis: Here's a concise section-by-section analysis of the paper, structured to match the actual organization of the text (noting where the full text cuts off):

---

### **1. Abstract**  
*Concise summary*: QwaveMPS is an open-source Python library for simulating one-dimensional quantum waveguide systems using matrix product states (MPS). It enables efficient, scalable simulations of non-Markovian waveguide-QED dynamics (e.g., time-delayed feedback, strong nonlinearities) by focusing computational resources on relevant system parts. This approach outperforms full Hilbert space methods in computational cost while treating quantized atoms and photons on equal footing.

---

### **2. Introduction**  
*Concise summary*: Waveguide-QED systems (e.g., atoms coupled to 1D photon fields) exhibit rich non-Markovian effects like time-delayed feedback and vacuum oscillations. Traditional methods (e.g., Markovian approximations, master equations) fail in strongly non-Markovian regimes, while advanced techniques (scattering theories) are limited to weak excitations. QwaveMPS addresses this gap using MPS-based simulations to model complex waveguide-QED dynamics without approximations.

---

### **3. Theoretical Background**  
*Concise summary*: The paper derives the waveguide-QED Hamiltonian (coupling emitters to quantized photons) and discretizes time into "bins" to handle non-Markovian effects. It explains how MPS formalism works:  
- Time-evolution operators are converted to matrix product operators (MPOs)  
- States are represented as tensor networks with "orthogonality centers" (OCs)  
- This enables efficient simulation of systems with up to *N* photons in the waveguide.

---

### **4. QwaveMPS Package**  
*Concise summary*: An open-source Python library designed to implement MPS-based simulations for waveguide-QED systems. Key features include:  
- User-friendly interface for constructing/evolving quantum states and operators  
- Handles Markovian *and* non-Markovian regimes (e.g., time-delayed feedback)  
- Supports strong nonlinearities and multi-photon dynamics  
- Targets quantum optics and circuit applications (e.g., superconducting qubits).

---

### **5. Examples & Applications**  
*Concise summary*: The paper demonstrates QwaveMPS capabilities through 6 illustrative cases (Fig. 1):  
- **Linear regime**: TLS decay in waveguides (Fig. 1a,b)  
- **Nonlinear regime**: Two-TLS decay (Fig. 1c,d)  
- **Classical drives**: Continuous-wave pumps/pulsed light (Fig. 1e)  
- **Quantum pulses**: Fock-state pulses (Fig. 1f)  
These showcase the tool’s ability to model diverse waveguide-QED phenomena without Markovian approximations.

---

### **6. Conclusion**  
*Inferred from text*: QwaveMPS bridges theoretical MPS methods with practical quantum optics applications by providing an accessible, open-source tool for non-Markovian waveguide-QED simulations. It overcomes limitations of existing frameworks (e.g., QuTiP’s Markovian focus) and enables scalable studies of time-delayed feedback, strong nonlinearities, and multi-photon dynamics in quantum circuits.

---

### Key Takeaways for the User
- **Why this matters**: Solves a critical gap in quantum simulation (non-Markovian waveguide-QED) where most tools fail.  
- **Practical value**: Open-source, efficient, and ready for real-world quantum circuit applications (e.g., superconducting systems).  
- **Limitation noted**: MPS is niche in quantum optics but addresses high-demand scenarios (e.g., time-delayed feedback).  
- **GitHub**: [github.com/SofiaArranzRegidor/QwaveMPS](https://github.com/SofiaArranzRegidor/QwaveMPS) (for implementation).

This analysis focuses *only* on the sections explicitly covered in the provided text (no extrapolation beyond the given content). The structure aligns with the paper’s actual flow: **Abstract → Intro → Theory → Package → Examples → Conclusion**.
            
[Read full paper](http://arxiv.org/abs/2602.15826v1)
#===================================================#



Paper: Ensemble-size-dependence of deep-learning post-processing methods that minimize an (un)fair score: motivating examples and a proof-of-concept solution
Authors: Christopher David Roberts
Categories: physics.ao-ph, cs.LG
Published: 2026-02-17 18:59 UTC
Source: Arxiv
            
Relevance Score: 0.654/1.00
            
Llm analysis: Here are concise, section-by-section summaries based on the provided preprint text (structured to match the paper's explicit sections and logical flow):

---

### **1. Abstract**  
*Concise Summary*: Fair scoring rules (e.g., adjusted CRPS) reward ensemble members behaving as independent draws from the true distribution. However, deep-learning post-processing methods that introduce structural dependencies between ensemble members (e.g., via linear calibration or transformers) can violate this assumption, causing ensemble-size-dependent unreliability (e.g., over-dispersion). The authors propose "trajectory transformers" as a proof-of-concept solution that maintains ensemble-size independence while improving reliability for ECMWF subseasonal forecasts (tested with 3 vs. 9 training members).

---

### **2. Introduction**  
*Concise Summary*: Ensemble forecasting approximates atmospheric states via Monte Carlo sampling, but raw ensembles often suffer from biases and unreliability. Fair scores (e.g., adjusted CRPS) are ideal loss functions for post-processing but require ensemble members to be exchangeable (independent draws from the same distribution). Distribution-aware methods (e.g., deep learning) that add dependencies between members can break this assumption, leading to systematic unreliability. The paper focuses on transformer-based post-processing (PoET framework) and proposes "trajectory transformers" to achieve ensemble-size independence.

---

### **3. An Idealized Example: Gaussian Forecasts and Linear Calibration**  
*Concise Summary*: Using Gaussian forecasts, the authors demonstrate how a linear member-by-member calibration (minimizing CRPS or aCRPS) introduces ensemble-size-dependent miscalibration. For finite ensembles, this method can cause over-dispersion (systematic unreliability) even when optimized for fair scores, highlighting the risk of structural dependencies between members.

---

### **4. ECMWF Subseasonal Forecast Data and Transformer Methods**  
*Concise Summary*: The study uses weekly mean 2-metre temperature (T₂m) forecasts from ECMWF’s subseasonal system. It compares two transformer-based methods:  
- **Ensemble Transformer**: Applies self-attention across ensemble dimension (introduces member dependencies).  
- **Trajectory Transformer**: Applies self-attention across *lead time* (not ensemble dimension), enabling trajectory-aware corrections without member dependencies. This design preserves aCRPS compatibility while being ensemble-size agnostic.

---

### **5. Results and Validation**  
*Concise Summary*: When applied to ECMWF T₂m forecasts, trajectory transformers:  
- Reduce systematic model biases.  
- Maintain or improve reliability regardless of training ensemble size (3 vs. 9 members) or real-time ensemble size (9 vs. 100 members).  
- Achieve ensemble-size independence without compromising fair scoring (aCRPS).

---

### **6. Conclusion**  
*Concise Summary*: Trajectory transformers provide a proof-of-concept solution for ensemble-size-independent deep-learning post-processing. By applying self-attention across lead time (not ensemble dimension), they eliminate structural dependencies between members, ensuring fair scoring (aCRPS) compatibility while improving reliability for both small and large ensembles. This approach addresses a critical limitation of existing transformer-based methods in operational forecasting.

---

### Key Insights from the Paper:
- **Core Problem**: Deep learning post-processing can violate the exchangeability assumption of fair scores (e.g., aCRPS), causing ensemble-size-dependent unreliability.  
- **Solution**: Trajectory transformers (vs. ensemble transformers) maintain conditional independence of members, enabling consistent reliability across ensemble sizes.  
- **Validation**: Works for ECMWF subseasonal T₂m forecasts with minimal bias and robust reliability (tested across 3–100 member ensembles).  

*Note: The preprint cuts off mid-sentence in Section 4, but the results are explicitly stated in the abstract and conclusion for the ECMWF application.*
            
[Read full paper](http://arxiv.org/abs/2602.15830v1)
#===================================================#



Paper: Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching
Authors: Zhen Wu, Xiaoyu Huang, Lujie Yang et al.
Categories: cs.RO, cs.AI, cs.LG
Published: 2026-02-17 18:59 UTC
Source: Arxiv
            
Relevance Score: 0.653/1.00
            
Llm analysis: Here's a concise section-by-section analysis of the paper, structured for clarity and research utility. I've focused on **only the sections explicitly covered in the provided text** (the paper cuts off mid-Section III, so I've omitted full Results/Conclusion sections where data is incomplete). Each summary is ≤4 lines and highlights *actionable insights* for researchers.

---

### 1. Abstract (Summary)
The paper introduces **Perceptive Humanoid Parkour (PHP)**, a framework enabling Unitree G1 robots to autonomously execute dynamic parkour skills (e.g., climbing 1.25m obstacles, vaulting at ~3 m/s) via onboard perception. It chains human motion skills using motion matching and distills multiple expert policies into a single depth-conditioned student policy, achieving zero-shot sim-to-real transfer for complex, contact-rich maneuvers.

---

### 2. Introduction (Summary)
Humanoid parkour requires **highly dynamic skills** (e.g., climbing above robot height), **perception-driven adaptation**, and **long-horizon skill chaining**—challenges exacerbated by scarce human motion data (few demonstrations per skill). PHP addresses this by: (1) retargeting human motions into robot-compatible atomic skills, (2) composing them via motion matching for diverse trajectories, and (3) distilling multiple RL-trained experts into a depth-conditioned policy for autonomous obstacle navigation.

---

### 3. Related Works (Summary)
- **Perceptive terrain traversal**: Quadrupeds excel at parkour courses but humanoids struggle with high-dimensional control; most use teacher-student pipelines (e.g., DAgger) for low-dynamic tasks.  
- **Human motion chaining**: Prior work (e.g., AMP, kinematics models) faces skill transition challenges in low-data regimes. Motion matching is underutilized in robotics despite its simplicity and effectiveness in animation.

---

### 4. Methods (Summary)
PHP uses a **modular pipeline**:  
1. **Skill composition**: Retarget human motions → atomic skills → motion matching (nearest-neighbor search) generates diverse long-horizon trajectories.  
2. **Teacher training**: RL-based motion tracking policies for each skill (using privileged states).  
3. **Distillation**: Hybrid DAgger + RL distills multiple teachers into a single depth-conditioned student policy for real-time obstacle selection.  
*Key innovation*: Motion matching enables smooth transitions without manual reward engineering.

---

### 5. Results (Inferred from Abstract & Figure 1)
- **Real-world validation** on Unitree G1:  
  - Climbed 1.25m obstacles (96% of robot height)  
  - Executed 60s continuous parkour with autonomous skill selection (vaulting, climbing, rolling)  
  - Zero-shot transfer from simulation to real robot with onboard depth sensing  
- *Critical metric*: Achieved high-speed maneuvers (3 m/s) with robust adaptation to obstacle perturbations.

---

### 6. Notes on Omitted Sections
- **Section III (Adaptive Parkour)**: Text cuts off mid-sentence ("The objective of this work is to enable..."). *Inferred focus*: Likely details the motion matching pipeline and real-world implementation.  
- **Conclusion**: Not fully provided, but the abstract implies PHP enables *scalable, perception-driven skill chaining* for humanoids without manual reward design.

---

### Key Takeaways for Researchers
| **Component**          | **Contribution**                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| **Motion Matching**     | Generates diverse trajectories from sparse human data (avoids manual reward design) |
| **Distillation**         | Hybrid DAgger + RL enables zero-shot sim-to-real transfer for dynamic skills      |
| **Perception**           | Onboard depth sensing for real-time obstacle selection (no external vision)       |
| **Hardware**             | Unitree G1 robot achieves 1.25m climbs and 60s parkour (industry-leading)        |

This analysis focuses on **actionable insights** for robotics researchers (e.g., motion matching for skill chaining, distillation for real-world deployment) while explicitly noting where the paper is incomplete. I avoided technical jargon where possible to prioritize clarity. Let me know if you'd like deeper dives into specific sections!
            
[Read full paper](http://arxiv.org/abs/2602.15827v1)
#===================================================#



Paper: CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing
Authors: Zarif Ikram, Arad Firouzkouhi, Stephen Tu et al.
Categories: cs.LG, cs.AI
Published: 2026-02-17 18:58 UTC
Source: Arxiv
            
Relevance Score: 0.611/1.00
            
Llm analysis: Here are concise, section-by-section summaries of the paper **"CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing"** based on the provided text:

---

### **1. Abstract**  
CrispEdit is a scalable, second-order LLM editing algorithm that preserves model capabilities during updates by enforcing capability preservation as a hard constraint. It projects edit updates onto the *low-curvature subspace* of the capability loss landscape using Bregman divergence, ensuring minimal degradation of general capabilities (e.g., reasoning, truthfulness) while achieving high edit success. Unlike prior methods, it avoids base-model convergence assumptions and scales efficiently via Kronecker-factored curvature approximations.

---

### **2. Introduction**  
LLM editing faces a critical challenge: *capability preservation*—edits that appear successful may silently degrade broader capabilities (e.g., via proxy hacking). Prior methods (e.g., parameter localization, representation constraints) often bake in unrealistic assumptions about edit structure, leading to poor real-world performance despite strong results in teacher-forced evaluations. CrispEdit addresses this by:  
- Treating capability preservation as an *explicit constraint* (not a soft penalty),  
- Using **low-curvature projections** to ensure edits move along "valleys" of the capability loss landscape,  
- Leveraging **Bregman divergence** to avoid base-model convergence requirements,  
- Enabling scalable implementation via **K-FAC** and **matrix-free projections**.  
The paper demonstrates CrispEdit achieves >90% edit success with <1% capability degradation across benchmarks (MMLU, GSM8K, etc.) on LLaMA-3 and Qwen models.

---

### **3. Model Editing Problem**  
Formalizes LLM editing as a constrained optimization problem:  
- **Goal**: Minimize edit loss (e.g., negative log-likelihood of target outputs) while preserving capabilities on a reference dataset \(D_{\text{cap}}\).  
- **Constraint**: Capability loss change \(d(L_{\text{cap}}(\theta), L_{\text{cap}}(\theta_0)) \leq \epsilon\) (where \(\epsilon\) is small).  
- **Key insight**: Standard approaches (e.g., Lagrangian relaxation) are computationally expensive when \(|D_{\text{cap}}| \gg |D_{\text{edit}}|\). CrispEdit avoids this by solving the constraint via *low-curvature projections* instead.

---

### **4. CrispEdit: Curvature-Restricted In-Situ Parameter Editing**  
**Core innovation**: Projects edit updates onto the *low-curvature subspace* of the capability loss landscape to ensure capability preservation.  
- **Why low-curvature?** Pretrained loss landscapes are highly anisotropic (sharp in few directions, flat in others). Low-curvature projections minimize capability disruption while optimizing edits.  
- **Bregman divergence**: Replaces Euclidean distance to compute capability preservation. Crucially, it yields the *exact Gauss-Newton Hessian* (GNH) even when the base model is **not trained to convergence**—a major advantage over prior methods.  
- **Scalability**: Uses **K-FAC** to approximate curvature efficiently and a **matrix-free projector** that:  
  (a) Rotates gradients into a Kronecker eigenbasis,  
  (b) Masks high-curvature components,  
  (c) Rotates back—avoiding massive matrix constructions.  
- **Theoretical link**: Proves AlphaEdit and Adam-NSCL are *restrictive special cases* of CrispEdit, explaining their weaker capability preservation.

---

### **5. Results**  
- **Small-scale**: On MNIST/FashionMNIST, low-curvature projections achieve strongest capability preservation with K-FAC approximations.  
- **LLM-scale**: On LLaMA-3-8B and Qwen-2.5-1.5B:  
  - **High edit success** (reliable autoregressive generations),  
  - **<1% capability degradation** across MMLU, GSM8K, IFEval, ARC, TruthQA,  
  - **Efficiency**: 3,000 edits in ~6 minutes on NVIDIA A40 GPU.  
- **Real-world robustness**: Works for *batch* and *sequential* editing without compromising out-of-scope knowledge or general skills.

---

### **6. Conclusion**  
CrispEdit solves the critical trade-off between edit efficacy and capability preservation by:  
1. Using **low-curvature projections** to enforce capability constraints,  
2. Leveraging **Bregman divergence** for GNH-based updates without convergence assumptions,  
3. Enabling **scalable implementation** via K-FAC and matrix-free projections.  
It outperforms prior editors in real-world LLM editing, maintaining >99% capability integrity while achieving high edit success—making it a practical solution for deploying LLMs in dynamic environments.

---

### Key Notes for Context:
- **Why "low-curvature" matters**: Most parameter updates move perpendicularly to capability contours (high degradation). Low-curvature paths follow "valleys" where capabilities change minimally (Fig 2).  
- **Bregman divergence advantage**: Standard Hessian-based methods require base models to converge—unrealistic for LLMs. Bregman divergence avoids this by using GNH.  
- **Real-world impact**: CrispEdit is the first method to achieve *both* high edit success and <1% capability degradation on diverse benchmarks (Fig 1).  
- **Date note**: The paper lists "February 17, 2026" (likely a placeholder; current year is 2024).

These summaries distill the paper’s technical contributions, methodology, and results while adhering to conciseness and research-focused clarity. Let me know if you'd like deeper dives into specific sections!
            
[Read full paper](http://arxiv.org/abs/2602.15823v1)
#===================================================#



Paper: Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation
Authors: Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki et al.
Categories: cs.RO, cs.CV, cs.LG
Published: 2026-02-17 18:59 UTC
Source: Arxiv
            
Relevance Score: 0.602/1.00
            
Llm analysis: Here's a concise section-by-section analysis of the Dex4D paper, based strictly on the provided text. I've identified the major structural sections and summarized each with key insights (avoiding speculation where text is incomplete).

---

### **1. Abstract**  
Dex4D is a framework that learns *task-agnostic* dexterous manipulation policies directly from simulation. It generates object-centric point tracks via video generation and 4D reconstruction, then uses these tracks to deploy a zero-shot "Anypose-to-Anypose" policy (manipulating any object to any pose) in the real world without fine-tuning. The policy generalizes across novel objects, scenes, and trajectories while requiring no real-world data collection.

### **2. Introduction**  
The paper addresses the bottleneck of *high-quality, diverse manipulation data* in real-world settings (expensive, error-prone collection). While sim-to-real reinforcement learning (RL) offers scalability, prior approaches require heavy engineering for task-specific rewards and environments. Dex4D proposes a *task-agnostic* solution: training a single "Anypose-to-Anypose" policy in simulation that can be *zero-shot deployed* via object-centric point tracks from generated videos. Key innovations include: (1) leveraging video generation for high-level planning, (2) using 4D reconstruction for point tracks as policy inputs, and (3) Paired Point Encoding for robust goal representation.

### **3. Related Work**  
- **Generalizable Dexterous Manipulation**: Prior works focus on grasping or in-hand tasks but lack autonomy for complex tasks.  
- **Video-Based Robot Learning**: Video generation models show promise as planners but suffer from embodiment gaps and lack closed-loop feedback.  
- **3D Policy Learning**: Existing methods use point clouds or scene representations but don’t extend goal-conditioned learning to *task-agnostic* policies.  
*Why Dex4D is novel*: Integrates video generation + 4D reconstruction for closed-loop control, enabling robust sim-to-real transfer without task-specific tuning.

### **4. Learning Point Track Policy via Task-Agnostic Sim-to-Real**  
*(Note: Text cuts off here, but the section describes the core methodology)*  
- **Anypose-to-Anypose (AP2AP)**: A task-agnostic MDP where the policy transforms *any* object from *any* initial pose to *any* target pose (no predefined grasps/motion primitives).  
- **Paired Point Encoding**: A novel goal representation that encodes *correspondences* between current and target object points (vs. separate encoding).  
- **Teacher-Student Learning**: Uses a teacher policy trained via RL with full object points (privileged states) and distills a student action world model from partial observations (robot proprioception + masked point tracks).  
- *Key insight*: The policy predicts actions and future states jointly via a transformer-based architecture, enabling real-time closed-loop control.

### **5. Results**  
*(Implied from abstract and related work)*  
- **Sim-to-real success**: Zero-shot deployment on real robots without fine-tuning.  
- **Generalization**: Strong performance on novel objects, scenes, backgrounds, and trajectories.  
- **Advantages over baselines**: Higher success rates, better robustness, and scalability compared to task-specific sim-to-real methods.  
*(Note: Full results section not provided in text)*

### **6. Conclusion**  
Dex4D enables *task-agnostic* dexterous manipulation via a single sim-trained policy that uses generated videos to extract point tracks for real-world execution. Its core contributions—AP2AP formulation, Paired Point Encoding, and teacher-student distillation—achieve zero-shot deployment and generalize across diverse configurations. This approach eliminates the need for task-specific simulation tuning, making dexterous manipulation scalable and robust.

---

### Key Observations from Analysis:
- **Why this matters**: Solves the critical scalability gap in dexterous manipulation by avoiding real-world data collection and task-specific rewards.  
- **Technical novelty**: Paired Point Encoding (not used in prior works) enables robust goal representation for high-DoF hands.  
- **Limitation in text**: The Methods section cuts off mid-sentence, so I focused on explicitly described components (AP2AP, Paired Point Encoding, teacher-student architecture).  
- **Real-world impact**: Zero-shot deployment on real robots with no fine-tuning—critical for practical robotics applications.

This summary covers all identifiable sections while staying strictly within the provided text. For full results details, the paper’s project page ([https://dex4d.github.io](https://dex4d.github.io)) would be the best reference.
            
[Read full paper](http://arxiv.org/abs/2602.15828v1)
#===================================================#



Papers similar in chroma_db to 'deep+computer+vision':

1) 6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems.
Similarity: 0.5515

2) QwaveMPS: An efficient open-source Python package for simulating non-Markovian waveguide-QED using matrix product states.
Similarity: 0.5453

3) The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context.
Similarity: 0.5407

4) Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment.
Similarity: 0.5296

#===================================================#



Additional takeaways:
Trending categories: cs.RO, quant-ph, cs.CV, cs.LG, physics.ao-ph, cs.AI
Sources count: arxiv: 5 papers

Most active fields: cs.RO

This post is AI-generated. 2026-02-19 03:48 UTC*