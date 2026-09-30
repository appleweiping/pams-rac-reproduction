{
  "schema_version": "1.0",
  "audit_id": "20260815_sol_canonical",
  "audit_type": "canonical_novelty_check",
  "as_of_date": "2026-08-15",
  "reviewer": {
    "reviewer_model": "gpt-5.6-sol",
    "reviewer_family": "openai",
    "independence": "same-family",
    "acceptance_status": "provisional",
    "role": "fresh zero-context ARIS novelty-check reviewer",
    "decision_rule": "A candidate advances only as a bounded experiment when its exact residual mechanism is not directly disclosed by the checked primary sources. Failure to find a disclosure is not proof of novelty, and component-wise novelty is rejected when the components are established."
  },
  "scope": {
    "venue": "ICASSP 2027",
    "research_problem": "Pose-driven multi-person repetitive action counting under non-stationary per-person tempo, with no per-person count or period labels used by the counting objective.",
    "audited_candidates": [
      "warp-equivariant-private-phase-flow",
      "masked-geometric-tssm-ridge-completion",
      "identity-private-state-isolation"
    ],
    "excluded_as_novelty_evidence": [
      "v46",
      "v62",
      "v63",
      "Missing implementations",
      "Missing experiments",
      "Missing search hits"
    ],
    "external_evidence_cutoff": "2026-08-15"
  },
  "input_ledger": [
    {
      "path": "RESEARCH_BRIEF.md",
      "sha256": "f5b9d75fc2415a03ba250279edf1e47af978341b7b9d7eef892f2d43a56b7c37"
    },
    {
      "path": "idea-stage/IDEA_REPORT.md",
      "sha256": "f7a03eb51f6ac429131fc0bde8f0d3bb6a684703a233bee3181a0836708113c3"
    },
    {
      "path": "idea-stage/CANDIDATE_UNION.json",
      "sha256": "4f3a85cc8773e63e0156acdd82befc9bad252f51980804ee418c49218132f22f"
    },
    {
      "path": "EXPERIMENT_AUDIT.md",
      "sha256": "2ac7fe4a1a8208287394a2ac4c1771aa0faf30e14319bef8144e509f876f0820"
    },
    {
      "path": "SERVER_METHOD_INVENTORY.md",
      "sha256": "0cda0704f64434296049d18c35e97fd267a0a7168300152e97b142d6ecc363a6"
    }
  ],
  "evidence_policy": {
    "allowed_external_sources": [
      "Official publisher or proceedings pages",
      "Official proceedings PDFs and supplements",
      "Author primary preprints",
      "Official author project pages and repositories"
    ],
    "disallowed_as_verdict_evidence": [
      "Search-result snippets without opening or cross-checking a primary source",
      "Secondary paper summaries",
      "Papers With Code",
      "ResearchGate mirrors",
      "Connected-paper graphs",
      "Unverified code branches or server version numbers"
    ],
    "novelty_standard": [
      "No 'first' claim is licensed.",
      "No missing-result inference is permitted.",
      "Established operators such as temporal self-similarity, masking, recurrent per-track state, time-warp augmentation, mixture routing, peak consensus, and normalized overlap-add are not independent contributions.",
      "A defensible claim must identify a narrow conjunction, compare against the closest mechanisms on a common protocol, and survive a kill ablation.",
      "Same-family review remains provisional even when the literature search finds a plausible residual."
    ]
  },
  "query_log": [
    {
      "query_id": "Q01",
      "date": "2026-08-15",
      "queries": [
        "MultiCounter Multi-person Repetitive Action Counting ECAI 2024 paper arXiv",
        "MultiCounter+ Spatial-Temporal Consistency Long-Short Period Awareness TCSVT 2026",
        "Count What Repeats Period-Adaptive Multi-Scale Consistency Self-Supervised Repetitive Action Counting PAMS CVPR 2026 supplement",
        "RepNet Counting Out Time CVPR 2020 official paper"
      ],
      "primary_sources_resolved": [
        "S01",
        "S02",
        "S03",
        "S04",
        "S05"
      ],
      "finding": "MultiCounter already defines joint multi-instance detection, tracking, period localization, and per-person counts. MultiCounter+ adds spatial-temporal consistency and long-short-period awareness. PAMS and RepNet occupy self-supervised or synthetic temporal-period representation territory."
    },
    {
      "query_id": "Q02",
      "date": "2026-08-15",
      "queries": [
        "site:ieeexplore.ieee.org/document 3670243 MultiCounter+",
        "\"MultiCounter+\" repetition counting",
        "site:arxiv.org MultiCounter+ repetition counting",
        "\"Spatial-Temporal Consistency\" \"Long-Short Period\" MultiCounter"
      ],
      "primary_sources_resolved": [
        "S02"
      ],
      "finding": "IEEE's record confirms MultiCounter+'s task-specific heads with spatial-temporal consistency, long-short-period awareness, MultiRep supervision, and large-scale synthetic pretraining."
    },
    {
      "query_id": "Q03",
      "date": "2026-08-15",
      "queries": [
        "HTRM-Net repetitive action counting TMM 2025 3535385 official",
        "\"Motion Feature Learning\" repetitive action counting pose",
        "\"Geometric Motion Feature Learning\" repetitive action counting",
        "JTSPS-Net joint-wise temporal self-similarity periodic selection TCSVT 2024"
      ],
      "primary_sources_resolved": [
        "S08",
        "S09",
        "S11"
      ],
      "finding": "Motion Feature Learning already uses a TSM reconstruction loss; HTRM-Net combines multiple TSSMs, random matrix dropping, local context, and adaptive multi-scale fusion; JTSPS occupies skeleton joint-wise TSS and coarse-to-fine periodic selection."
    },
    {
      "query_id": "Q04",
      "date": "2026-08-15",
      "queries": [
        "site:ieeexplore.ieee.org \"JTSPS-Net\"",
        "site:arxiv.org \"JTSPS-Net\" repetitive",
        "\"Joint-Wise Temporal Self-Similarity\" repetition counting",
        "\"Joint-wise Temporal Self-similarity Periodic Selection\""
      ],
      "primary_sources_resolved": [
        "S11"
      ],
      "finding": "The official IEEE record identifies supervised skeleton input, joint-wise TSS periodic selection, temporal multi-scale fusion, and impulse-map regression using one labeled frame per action unit."
    },
    {
      "query_id": "Q05",
      "date": "2026-08-15",
      "queries": [
        "SimPer self-supervised periodic representation learning paper",
        "CycleCL cycle consistent learning repetition counting paper",
        "time-equivariant learning self-supervised video time warp equivariance paper",
        "temporal equivariance time warping representation learning video"
      ],
      "primary_sources_resolved": [
        "S12",
        "S13",
        "S14"
      ],
      "finding": "SimPer already uses periodicity-varying and invariant augmentations with continuous frequency-aware contrast; CycleCL explicitly learns phase-sensitive, repetition-invariant features; time-equivariant contrastive video learning preserves temporal transformations rather than discarding them."
    },
    {
      "query_id": "Q06",
      "date": "2026-08-15",
      "queries": [
        "masked skeleton modeling self-supervised pose motion representation official paper",
        "SkeletonMAE masked autoencoder skeleton action recognition paper",
        "MAMP masked motion predictors skeleton action recognition ICCV 2023 official",
        "MotionBERT masked motion modeling ICCV 2023 official"
      ],
      "primary_sources_resolved": [
        "S15",
        "S16",
        "S17"
      ],
      "finding": "Spatial-temporal skeleton masking, topology-guided joint/edge reconstruction, masked motion prediction, and recovery from noisy partial 2D pose are established self-supervised motion-representation devices."
    },
    {
      "query_id": "Q07",
      "date": "2026-08-15",
      "queries": [
        "multi object tracking per object memory transformer official paper MeMOT",
        "MOTR track query memory multi object tracking ECCV 2022 official",
        "MeMOT long-term memory multi-object tracking CVPR 2022 official",
        "TrackFormer track queries autoregressive identity multi object tracking official"
      ],
      "primary_sources_resolved": [
        "S18",
        "S19",
        "S20"
      ],
      "finding": "Identity-preserving track queries, per-object spatio-temporal memories, and recurrently updated track state are established in multi-object tracking. State isolation alone is not a new architecture."
    },
    {
      "query_id": "Q08",
      "date": "2026-08-15",
      "queries": [
        "site:openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats PAMS-TCC consensus experts period adaptive multi scale",
        "site:openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats speed variation stochastic piecewise PAMS",
        "\"PAMS-TCC\" repetitive action counting",
        "\"stochastic perturbation\" PAMS RepCount"
      ],
      "primary_sources_resolved": [
        "S03",
        "S04"
      ],
      "finding": "The PAMS supplement already reports internal tempo fluctuation: a random 5-7 second segment is resampled by a factor in [0.5, 1.5], repeated ten times for each of 152 videos. Therefore, localized within-video speed stress is not an unoccupied evaluation idea."
    },
    {
      "query_id": "Q09",
      "date": "2026-08-15",
      "queries": [
        "\"TWCRAC\" \"Time-Window-Cycle\"",
        "\"TWCRAC\" \"dynamic threshold\" \"local\" periodicity",
        "\"TWCRAC\" pdf",
        "site:ssrn.com/abstract=6630216 TWCRAC"
      ],
      "primary_sources_resolved": [
        "S06"
      ],
      "finding": "The accessible SSRN record discloses intra-video TCC, local-feature statistical thresholds, pose input, non-stationary motion, and video-level labels. The full 31-page method could not be reliably extracted beyond the SSRN-hosted abstract, so exact architectural overlap remains unresolved."
    },
    {
      "query_id": "Q10",
      "date": "2026-08-15",
      "queries": [
        "site:ieeexplore.ieee.org/document/ \"Repetitive Action Counting With Hybrid Temporal Relation Modeling\"",
        "\"Hybrid Temporal Relation Modeling\" TSSM random matrix dropping local temporal context",
        "\"HTRM-Net\" \"random matrix dropping\"",
        "\"HTRM-Net\" \"adaptive multi-scale\" repetitive"
      ],
      "primary_sources_resolved": [
        "S08"
      ],
      "finding": "HTRM-Net directly narrows any claim based on masked or dropped TSSMs, local temporal context, multi-scale correlation, interruption handling, or non-uniform periods."
    },
    {
      "query_id": "Q11",
      "date": "2026-08-15",
      "queries": [
        "site:ieeexplore.ieee.org/document/10647309 \"Rethinking Temporal Self-Similarity\"",
        "site:ieeexplore.ieee.org/document/10534255 \"Joint-Wise Temporal Self-Similarity\"",
        "site:arxiv.org/abs/2407.09431 \"reference-TSM consistency\"",
        "site:arxiv.org/abs/2407.09431 \"action start\""
      ],
      "primary_sources_resolved": [
        "S10",
        "S11"
      ],
      "finding": "Rethinking TSM already uses a reference-TSM consistency loss and full-resolution action-start response, while JTSPS supplies the skeleton/joint-wise TSS and multi-scale boundary."
    },
    {
      "query_id": "Q12",
      "date": "2026-08-15",
      "queries": [
        "DeepPhase periodic autoencoders learning motion phase manifolds official paper",
        "phase function neural network unit circle phase motion reconstruction time warp equivariant",
        "local phase representation human motion periodic autoencoder",
        "learning phase variables periodic motion self supervised"
      ],
      "primary_sources_resolved": [
        "S21"
      ],
      "finding": "DeepPhase is a missing close prior: it learns multi-channel local periodic phase manifolds from unstructured full-body motion without labels and reconstructs motion. It does not perform MRAC or the proposed explicit warp-derivative law, but it blocks novelty claims for unsupervised motion phase itself."
    },
    {
      "query_id": "Q13",
      "date": "2026-08-15",
      "queries": [
        "site:openaccess.thecvf.com/content/WACV2024/papers/Li_Repetitive_Action_Counting \"Matrix Reconstruction Loss\"",
        "\"Repetitive Action Counting With Motion Feature Learning\" \"reconstruction loss\" temporal self similarity matrix",
        "site:openaccess.thecvf.com/content/WACV2024/papers/Li_Repetitive_Action_Counting \"reconstructed\" TSM",
        "site:openaccess.thecvf.com/content/WACV2024/supplemental/Li_Repetitive_Action_Counting matrix reconstruction"
      ],
      "primary_sources_resolved": [
        "S09"
      ],
      "finding": "Motion Feature Learning reconstructs TSM structure using a ground-truth matrix derived from annotated repetition boundaries. It is not label-free, but it establishes matrix reconstruction as a RAC mechanism and demands a direct supervision-matched ablation."
    }
  ],
  "source_ledger": [
    {
      "source_id": "S01",
      "work": "MultiCounter: Multiple Action Agnostic Repetition Counting in Untrimmed Videos",
      "status": "ECAI 2024 author primary preprint",
      "primary_url": "https://arxiv.org/abs/2409.04035",
      "accessed": "2026-08-15",
      "evidence_scope": "MRAC task definition, instance queries, mixed spatial-temporal interaction, per-frame period velocity and periodicity, per-instance localization and count output, MultiRep, Period-AP.",
      "novelty_effect": "Blocks claims for first MRAC, first per-person count, first asynchronous or variable-period multi-person counting, and first instance-indexed temporal modeling."
    },
    {
      "source_id": "S02",
      "work": "MultiCounter+: Toward Efficient and Robust Multi-Instance Repetitive Action Counting",
      "status": "IEEE TCSVT 2026 official publisher record",
      "primary_url": "https://ieeexplore.ieee.org/document/11421441/",
      "accessed": "2026-08-15",
      "evidence_scope": "Spatial-temporal consistency, long-short period awareness, synthetic pretraining, unified multi-person detection/tracking/counting.",
      "novelty_effect": "Blocks identity-consistency, long/short-period awareness, robust variable-speed MRAC, and unified multi-person counting as standalone novelty."
    },
    {
      "source_id": "S03",
      "work": "Count What Repeats: Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action Counting",
      "status": "CVPR 2026 Findings official proceedings main paper",
      "primary_url": "https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html",
      "accessed": "2026-08-15",
      "evidence_scope": "Unlabeled skeleton sequences, period-adaptive multi-scale TCC, intra-video phase alignment, cross-video separation, inference-time multi-expert peak consensus.",
      "novelty_effect": "Blocks self-supervised pose periodic learning, period-adaptive multi-scale consistency, and generic multi-expert consensus claims."
    },
    {
      "source_id": "S04",
      "work": "PAMS Supplementary Material",
      "status": "CVPR 2026 Findings official supplement",
      "primary_url": "https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Internal tempo fluctuation by random 5-7 second segment resampling in [0.5x, 1.5x], ten repetitions over 152 RepCount test videos, plus global speed robustness.",
      "novelty_effect": "Blocks any claim that localized within-video tempo corruption or piecewise speed stress is absent from prior PAMS evaluation."
    },
    {
      "source_id": "S05",
      "work": "Counting Out Time: Class Agnostic Video Repetition Counting in the Wild (RepNet)",
      "status": "CVPR 2020 official proceedings",
      "primary_url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html",
      "accessed": "2026-08-15",
      "evidence_scope": "Temporal self-similarity bottleneck, synthetic repetitions, per-frame period-length and periodicity predictions.",
      "novelty_effect": "Blocks TSSM, synthetic temporal resampling, and time-varying frame-level period/periodicity as standalone contributions."
    },
    {
      "source_id": "S06",
      "work": "TWCRAC: Leveraging Temporal Cycle Consistency and Dynamic Thresholding for Robust Repetitive Action Counting",
      "status": "SSRN 2026 author preprint record; abstract-level method evidence",
      "primary_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
      "accessed": "2026-08-15",
      "evidence_scope": "Time-window-cycle framing, intra-video TCC, pose, video-level labels, dynamic statistical thresholds from local periodicity features, non-stationary motion.",
      "novelty_effect": "Creates the highest unresolved collision risk for local windows plus cycle consistency plus adaptive local tempo counting."
    },
    {
      "source_id": "S07",
      "work": "TWCRAC SSRN-hosted PDF delivery",
      "status": "SSRN primary PDF endpoint; extraction exposed only abstract-level text",
      "primary_url": "https://papers.ssrn.com/sol3/Delivery.cfm/929069d2-a143-4b2f-b90f-e36010c363c9-MECA.pdf?abstractid=6630216&mirid=1",
      "accessed": "2026-08-15",
      "evidence_scope": "Confirms the 31-page preprint exists, but did not permit reliable full-method comparison in this audit.",
      "novelty_effect": "Requires conservative unresolved-overlap language; absence of an extracted detail cannot support novelty."
    },
    {
      "source_id": "S08",
      "work": "Repetitive Action Counting with Hybrid Temporal Relation Modeling (HTRM-Net)",
      "status": "IEEE TMM 2025 author primary preprint",
      "primary_url": "https://arxiv.org/abs/2412.07233",
      "accessed": "2026-08-15",
      "evidence_scope": "Bi-modal TSSMs, random matrix dropping, local temporal context, adaptive multi-scale matrix fusion, non-uniform periods, interruption robustness.",
      "novelty_effect": "Blocks TSSM diversity, masking/dropping, local context, adaptive multi-scale temporal relation modeling, and interruption robustness as standalone novelty."
    },
    {
      "source_id": "S09",
      "work": "Repetitive Action Counting With Motion Feature Learning",
      "status": "WACV 2024 official proceedings",
      "primary_url": "https://openaccess.thecvf.com/content/WACV2024/html/Li_Repetitive_Action_Counting_With_Motion_Feature_Learning_WACV_2024_paper.html",
      "accessed": "2026-08-15",
      "evidence_scope": "RGB/flow branches, TSM reconstruction loss, density prediction; the ground-truth TSM is constructed from annotated repetition boundaries.",
      "novelty_effect": "Blocks generic TSM reconstruction and motion/TSM consistency claims, while leaving supervision-free pseudo-target construction as only a narrow possible distinction."
    },
    {
      "source_id": "S10",
      "work": "Rethinking Temporal Self-Similarity for Repetitive Action Counting",
      "status": "ICIP 2024 author primary preprint",
      "primary_url": "https://arxiv.org/abs/2407.09431",
      "accessed": "2026-08-15",
      "evidence_scope": "Reference-TSM consistency loss, full-resolution frame embeddings, action-start probability response, count from response.",
      "novelty_effect": "Blocks reference self-similarity consistency, full-resolution response reconstruction, and response-then-count as standalone novelty."
    },
    {
      "source_id": "S11",
      "work": "Joint-Wise Temporal Self-Similarity Periodic Selection Network for Repetitive Fitness Action Counting (JTSPS-Net)",
      "status": "IEEE TCSVT 2024 official publisher record",
      "primary_url": "https://ieeexplore.ieee.org/document/10534255",
      "accessed": "2026-08-15",
      "evidence_scope": "Skeleton input, joint-wise temporal self-similarity, coarse-to-fine periodic selection, temporal multi-scale fusion, supervised impulse-map regression.",
      "novelty_effect": "Blocks joint-wise or geometric skeleton TSSM and multi-scale periodic selection as standalone novelty."
    },
    {
      "source_id": "S12",
      "work": "SimPer: Simple Self-Supervised Learning of Periodic Targets",
      "status": "ICLR 2023 official author project and OpenReview-linked paper",
      "primary_url": "https://simper.csail.mit.edu/",
      "accessed": "2026-08-15",
      "evidence_scope": "Periodicity-invariant and periodicity-varying temporal augmentations, periodic feature similarity, soft frequency-regression contrastive learning.",
      "novelty_effect": "Blocks self-supervised speed augmentation and explicit continuous-frequency periodic representation as standalone novelty."
    },
    {
      "source_id": "S13",
      "work": "CycleCL: Self-Supervised Learning for Periodic Videos",
      "status": "WACV 2024 official proceedings paper",
      "primary_url": "https://openaccess.thecvf.com/content/WACV2024/papers/Destro_CycleCL_Self-Supervised_Learning_for_Periodic_Videos_WACV_2024_paper.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Self-supervised periodic-video features that are sensitive to cycle phase and invariant to repetition identity.",
      "novelty_effect": "Blocks phase-sensitive, repetition-invariant self-supervised features as standalone novelty."
    },
    {
      "source_id": "S14",
      "work": "Time-Equivariant Contrastive Video Representation Learning",
      "status": "ICCV 2021 official proceedings paper",
      "primary_url": "https://openaccess.thecvf.com/content/ICCV2021/papers/Jenni_Time-Equivariant_Contrastive_Video_Representation_Learning_ICCV_2021_paper.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Contrastive representations equivariant to relative temporal transformations, with ordered/overlapping/unordered clip objectives.",
      "novelty_effect": "Blocks temporal-transform equivariance in video representation learning as a generic contribution."
    },
    {
      "source_id": "S15",
      "work": "SkeletonMAE: Spatial-Temporal Masked Autoencoders for Self-supervised Skeleton Action Recognition",
      "status": "Author primary preprint",
      "primary_url": "https://arxiv.org/abs/2209.02399",
      "accessed": "2026-08-15",
      "evidence_scope": "Frame-level and joint-level spatial-temporal masking and skeleton reconstruction.",
      "novelty_effect": "Blocks skeleton masking and masked joint/frame reconstruction as standalone novelty."
    },
    {
      "source_id": "S16",
      "work": "SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training",
      "status": "ICCV 2023 official proceedings paper",
      "primary_url": "https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Topology-guided masked joints and edges, graph encoder-decoder, fine-grained spatial-temporal skeleton dependencies.",
      "novelty_effect": "Blocks topology-aware masked geometric skeleton reconstruction as standalone novelty."
    },
    {
      "source_id": "S17",
      "work": "Masked Motion Predictors are Strong 3D Action Representation Learners (MAMP)",
      "status": "ICCV 2023 official proceedings paper",
      "primary_url": "https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Masked spatio-temporal skeleton input, masked joint-motion prediction, motion-aware masking.",
      "novelty_effect": "Blocks masked joint-velocity or motion prediction as a generic self-supervised contribution."
    },
    {
      "source_id": "S18",
      "work": "MeMOT: Multi-Object Tracking With Memory",
      "status": "CVPR 2022 official proceedings",
      "primary_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html",
      "accessed": "2026-08-15",
      "evidence_scope": "Large spatio-temporal memory storing identity embeddings for each tracked object, per-object memory encoding, long-gap relinking.",
      "novelty_effect": "Blocks per-object identity memory and long-term identity state as standalone novelty."
    },
    {
      "source_id": "S19",
      "work": "TrackFormer: Multi-Object Tracking with Transformers",
      "status": "CVPR 2022 official proceedings paper",
      "primary_url": "https://openaccess.thecvf.com/content/CVPR2022/papers/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Autoregressive identity-preserving track queries updated frame by frame.",
      "novelty_effect": "Blocks identity-indexed recurrent query state as a generic architecture."
    },
    {
      "source_id": "S20",
      "work": "MOTR: End-to-End Multiple-Object Tracking with Transformer",
      "status": "ECCV 2022 official implementation and author primary paper",
      "primary_url": "https://github.com/megvii-research/MOTR",
      "accessed": "2026-08-15",
      "evidence_scope": "Each track query models an object's entire track and is transferred and updated frame by frame.",
      "novelty_effect": "Reinforces that identity-specific persistent state is established tracking machinery."
    },
    {
      "source_id": "S21",
      "work": "DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds",
      "status": "ACM TOG / SIGGRAPH 2022 author-hosted primary paper",
      "primary_url": "https://i.cs.hku.hk/~taku/deepphase.pdf",
      "accessed": "2026-08-15",
      "evidence_scope": "Unsupervised periodic autoencoder, multi-dimensional local phase channels, frequency/amplitude/offset/phase, motion reconstruction from unstructured full-body motion.",
      "novelty_effect": "Blocks unsupervised local motion-phase manifolds and phase-conditioned reconstruction as standalone novelty."
    },
    {
      "source_id": "S22",
      "work": "MotionBERT: A Unified Perspective on Learning Human Motion Representations",
      "status": "ICCV 2023 official author project",
      "primary_url": "https://motionbert.github.io/",
      "accessed": "2026-08-15",
      "evidence_scope": "Recovery of 3D motion from noisy partial 2D observations and reusable geometric, kinematic, physical motion representation.",
      "novelty_effect": "Blocks recovery from masked/noisy pose and general pose-motion pretraining as standalone novelty."
    }
  ],
  "corrections": [
    {
      "correction_id": "C01",
      "severity": "material",
      "supplied_statement": "PAMS speed variation is only a whole-video global resampling test.",
      "corrected_statement": "PAMS's official supplement also contains an internal tempo fluctuation experiment: a random 5-7 second segment is sped up or slowed down with a multiplier uniformly sampled from [0.5, 1.5], repeated ten times for each of 152 RepCount test videos.",
      "source_ids": [
        "S04"
      ],
      "consequence": "Piecewise or localized within-video time-warp evaluation is required but cannot be claimed as novel. The proposed diagnostic must be stronger and predeclared, for example multiple monotone segments, acceleration and deceleration profiles, identity-specific perturbations, clean-corrupt pairing, and phase/routing traces."
    },
    {
      "correction_id": "C02",
      "severity": "material",
      "supplied_omission": "DeepPhase was absent from the closest-prior set for warp-equivariant-private-phase-flow.",
      "corrected_statement": "DeepPhase already learns unsupervised multi-channel local phase manifolds from full-body motion and reconstructs motion through phase/frequency/amplitude/offset features.",
      "source_ids": [
        "S21"
      ],
      "consequence": "The candidate cannot claim unsupervised phase, local motion phase, unit-circle-like periodic state, or phase-conditioned reconstruction. Only the exact MRAC conjunction and explicit warp-derivative transformation law remain potentially distinguishable."
    },
    {
      "correction_id": "C03",
      "severity": "important",
      "supplied_ambiguity": "Motion Feature Learning was summarized as TSM reconstruction without a clear supervision boundary.",
      "corrected_statement": "Its reference/ground-truth TSM is constructed from annotated repetition start and end points, so it is supervised rather than a count/period-label-free masked pseudo-target.",
      "source_ids": [
        "S09"
      ],
      "consequence": "The masked geometric candidate retains a narrow supervision distinction, but TSM reconstruction itself is established and cannot carry novelty."
    },
    {
      "correction_id": "C04",
      "severity": "important",
      "supplied_ambiguity": "Identity-private state was framed as a possible model contribution without fully importing the tracking-memory literature.",
      "corrected_statement": "MeMOT, TrackFormer, and MOTR already maintain identity-specific object memories or autoregressive track queries.",
      "source_ids": [
        "S18",
        "S19",
        "S20"
      ],
      "consequence": "Private state allocation is routine tracking architecture. The only plausible contribution is a repetition-counting-specific causal isolation diagnostic and evidence that state prevents cross-person count interference."
    },
    {
      "correction_id": "C05",
      "severity": "important",
      "supplied_ambiguity": "PAMS's experts could be confused with trainable tempo-specialized response functions.",
      "corrected_statement": "PAMS's cited multi-expert mechanism is an inference-time consensus over peak/window/smoothing configurations and scales; it is not evidence of learned identity-window response experts.",
      "source_ids": [
        "S03"
      ],
      "consequence": "Learned response experts remain technically distinguishable only if their specialization, routing dependence, and benefit over PAMS consensus are demonstrated. The word 'expert' alone is not a novelty boundary."
    },
    {
      "correction_id": "C06",
      "severity": "important",
      "supplied_ambiguity": "TWCRAC was grouped with label-free/self-supervised approaches without a precise supervision qualifier.",
      "corrected_statement": "The accessible SSRN abstract says its intra-video TCC uses only video-level labels, not no labels.",
      "source_ids": [
        "S06"
      ],
      "consequence": "The precise project wording must remain: the counting objective uses no per-person count or period labels, while detector, pose, tracker, augmentation correspondences, and any video-level supervision are separately disclosed."
    }
  ],
  "candidate_verdicts": {
    "warp-equivariant-private-phase-flow": {
      "verdict": {
        "code": "ADVANCE_BOUNDED_PILOT_NOT_NOVELTY_CLEARED",
        "status": "same-family provisional",
        "confidence": "medium-low",
        "reason": "No checked primary source directly disclosed the full conjunction of supplied per-person pose tracks, identity-private recurrent phase state, an explicit local warp-derivative law for signed phase increments, reversal sign handling, capacity-limited residual reconstruction, and one-decode-per-identity MRAC. However, every major component except that exact conjunction is occupied, and TWCRAC full-method overlap remains unresolved."
      },
      "overlap": [
        {
          "prior": "PAMS main and supplement",
          "source_ids": [
            "S03",
            "S04"
          ],
          "collision": "Self-supervised skeleton periodic representation, multi-scale period adaptation, intra-video phase consistency, inference consensus, and localized internal tempo perturbation."
        },
        {
          "prior": "RepNet",
          "source_ids": [
            "S05"
          ],
          "collision": "Synthetic temporal repetition, per-frame period and periodicity, and temporal self-similarity bottleneck."
        },
        {
          "prior": "TWCRAC",
          "source_ids": [
            "S06",
            "S07"
          ],
          "collision": "Time-window-cycle framing, intra-video TCC, pose, local statistical adaptation, and non-stationary motion; exact full-text boundary is unresolved."
        },
        {
          "prior": "SimPer",
          "source_ids": [
            "S12"
          ],
          "collision": "Self-supervised periodic representations from speed/periodicity augmentations and continuous frequency-aware contrast."
        },
        {
          "prior": "CycleCL",
          "source_ids": [
            "S13"
          ],
          "collision": "Phase-sensitive and repetition-invariant self-supervised periodic features."
        },
        {
          "prior": "Time-Equivariant Contrastive Video Representation Learning",
          "source_ids": [
            "S14"
          ],
          "collision": "Temporal-transformation equivariance as a representation-learning principle."
        },
        {
          "prior": "DeepPhase",
          "source_ids": [
            "S21"
          ],
          "collision": "Unsupervised local periodic phase channels and motion reconstruction from phase/frequency/amplitude/offset."
        },
        {
          "prior": "MultiCounter and MultiCounter+",
          "source_ids": [
            "S01",
            "S02"
          ],
          "collision": "Identity-indexed MRAC, variable-period actions, temporal consistency, per-person outputs, and period-aware multi-person counting."
        }
      ],
      "residual": {
        "narrow_claim_candidate": "A testable mechanism for MRAC in which known monotone local time warps supervise the transformation law of signed, window-local phase increments inside identity-private state, followed by per-track response reconstruction and one final decode, while the counting objective reads no per-person count or period labels.",
        "why_only_residual": [
          "The explicit derivative-scaled signed phase-increment law was not found in the checked RAC, periodic SSL, motion-phase, or temporal-equivariance sources.",
          "The use of separate physical-person state connects the phase law to asynchronous MRAC rather than generic single-sequence periodic representation.",
          "The residual is a conjunction claim, not a claim that phase, equivariance, recurrent state, time warps, pose, or reconstruction is new."
        ],
        "unresolved_threats": [
          "TWCRAC's inaccessible full method may contain a closer window-local tempo formulation.",
          "DeepPhase substantially narrows the phase representation boundary.",
          "PAMS's internal tempo fluctuation test removes the cleanest proposed evaluation novelty.",
          "A derivative-law loss may be mathematically correct yet empirically redundant with ordinary speed augmentation or SimPer-style frequency contrast.",
          "Physical phase may be non-identifiable for interrupted, non-cyclic, multi-limb, or aliased motion."
        ],
        "allowed_language_if_and_only_if_supported": "We study whether an explicit local time-warp transformation law for identity-indexed phase increments improves pose-driven MRAC under within-track tempo drift when the counting objective uses no per-person count or period labels."
      },
      "forbidden_claims": [
        "first multi-person repetition counter",
        "first per-person or identity-aware repetition counter",
        "first asynchronous or variable-speed repetition counter",
        "first pose-driven repetition counter",
        "first self-supervised periodic representation",
        "first phase representation for human motion",
        "first time-equivariant video representation",
        "first localized or piecewise speed perturbation test",
        "first window-local temporal cycle consistency method",
        "annotation-free",
        "end-to-end",
        "constant-time",
        "learned experts are novel merely because PAMS experts are not learned",
        "normalized overlap-add is a contribution by itself",
        "novel or state of the art before protocol-matched eligible evidence"
      ],
      "must_run": [
        "Freeze a checksum-verified source-disjoint MultiRep split and evaluator; expose no validation labels to training or hyperparameter search and access no sealed test set.",
        "Implement a dataloader audit proving that per-person count, period, cycle-boundary, and density fields never enter the counting objective; separately disclose pose, detector, tracker, augmentation-correspondence, and any video-level supervision.",
        "Compare on identical supplied tracks against PAMS main, PAMS with its supplementary internal-tempo corruption, RepNet, a Track-PAMS adaptation, SimPer-style speed contrast, CycleCL-style phase learning, a DeepPhase-style periodic autoencoder, and a time-equivariant contrastive baseline.",
        "Ablate the explicit warp-derivative scaling while preserving the same augmentations, remove reversal sign supervision, replace piecewise warps with global resampling, and replace signed phase increments with direct frequency regression.",
        "Ablate identity-private recurrence against stateless windows, shared scene state, state reset at every window, and track-global phase estimation.",
        "Test timestamp and augmentation-parameter shortcuts by removing pose input, shuffling pose within matched timestamps, randomizing crop origins, hiding absolute time, and predicting warp parameters from nuisance metadata.",
        "Use natural and synthetic non-stationarity: multiple monotone segments, smooth acceleration/deceleration, abrupt changes, pauses, interruptions, aliasing, and unrelated motion; report clean-to-corrupt paired degradation.",
        "Plot per-identity local phase increment, phase, confidence, response, routing if any, and reconstruction across window boundaries; verify positive phase mass is neither duplicated nor lost.",
        "Require three independent seeds, mean, sample standard deviation, and paired source-cluster bootstrap confidence intervals for Period-mAP, AP50, AP75, AvgMAE, and AvgOBO.",
        "Kill the phase-flow claim if the derivative-law ablation matches it within uncertainty, if a DeepPhase/SimPer/CycleCL baseline matches it, if timestamp-only retains over 20 percent of the gain, or if it fails the predeclared collapse and effective-rank gates.",
        "Obtain and inspect the complete TWCRAC paper before manuscript claim freeze; until then state the overlap as unresolved."
      ]
    },
    "masked-geometric-tssm-ridge-completion": {
      "verdict": {
        "code": "REJECT_AS_PRIMARY_NOVELTY_RETAIN_AS_CONTROL",
        "status": "same-family provisional",
        "confidence": "medium-high",
        "reason": "The candidate is an assembly of established TSSM, matrix reconstruction, skeleton geometry, masking, masked motion modeling, multi-scale periodic selection, local context, and response decoding. Its label-free cross-view masked off-diagonal completion is a narrow implementation distinction, not presently a persuasive primary contribution."
      },
      "overlap": [
        {
          "prior": "RepNet",
          "source_ids": [
            "S05"
          ],
          "collision": "Temporal self-similarity as the central periodic representation."
        },
        {
          "prior": "Motion Feature Learning",
          "source_ids": [
            "S09"
          ],
          "collision": "TSM reconstruction loss for repetitive action counting; supervision differs, but reconstruction does not."
        },
        {
          "prior": "HTRM-Net",
          "source_ids": [
            "S08"
          ],
          "collision": "Multiple TSSM views, random whole-matrix/channel dropping, local temporal context, and adaptive multi-scale fusion."
        },
        {
          "prior": "Rethinking Temporal Self-Similarity",
          "source_ids": [
            "S10"
          ],
          "collision": "Self-similarity consistency loss and full-resolution response-based counting."
        },
        {
          "prior": "JTSPS-Net",
          "source_ids": [
            "S11"
          ],
          "collision": "Skeleton joint-wise TSSM, coarse-to-fine periodic selection, and temporal multi-scale processing."
        },
        {
          "prior": "SkeletonMAE, graph SkeletonMAE, MAMP, and MotionBERT",
          "source_ids": [
            "S15",
            "S16",
            "S17",
            "S22"
          ],
          "collision": "Spatial-temporal skeleton masking, topology/geometric reconstruction, masked motion/velocity prediction, and recovery from partial noisy pose."
        },
        {
          "prior": "PAMS",
          "source_ids": [
            "S03",
            "S04"
          ],
          "collision": "Self-supervised skeleton periodic representation, multi-scale consistency, and tempo robustness."
        }
      ],
      "residual": {
        "narrow_claim_candidate": "A supervision distinction remains for completing masked off-diagonal self-similarity ridges from a different geometric pose view, using only within-track pose-derived pseudo-targets and excluding per-person count and period labels.",
        "why_insufficient_for_primary_novelty": [
          "Masked completion is a predictable combination of masked skeleton modeling and existing TSM reconstruction.",
          "Cross-view geometry and off-diagonal masking are design choices unless a new principle or consistent advantage over direct masked motion and TSM baselines is established.",
          "Ridge decoding, normalized overlap-add, and one per-identity decode are established or generic downstream operators.",
          "The candidate currently has no eligible implementation or result."
        ],
        "permitted_role": "Cheap negative control or fallback pretext in the same experimental framework, not the paper's novelty anchor.",
        "allowed_language_if_and_only_if_supported": "We evaluate cross-view masked off-diagonal TSSM completion as a label-restricted pretext for pose-track RAC."
      },
      "forbidden_claims": [
        "first masked TSSM method",
        "first skeleton masking method",
        "first TSSM reconstruction for repetition counting",
        "first joint-wise or geometric self-similarity counter",
        "first local-context or multi-scale TSSM counter",
        "matrix ridge completion is novel by itself",
        "multi-geometric views are novel by themselves",
        "label-free without disclosing externally supervised pose/tracking and pose-derived pseudo-targets",
        "one-decode-per-track or normalized overlap-add is novel",
        "primary method novelty based only on combining known blocks",
        "state of the art before eligible matched results"
      ],
      "must_run": [
        "Retain the identical encoder, windowing, decoder, tracks, augmentations, and compute budget across all pretext comparisons.",
        "Compare against constant, row mean, column mean, low-rank interpolation, bicubic/block interpolation, direct pose reconstruction, masked joint reconstruction, MAMP-style masked velocity prediction, supervised Motion Feature Learning when annotations are allowed as an oracle, RepNet TSSM, Rethinking-TSM consistency, JTSPS-style joint TSS, HTRM-style random matrix dropping, and PAMS.",
        "Ablate off-diagonal exclusion, each mask family, cross-view versus same-view target, handcrafted geometry families, ridge decoder, and the cross-view agreement distillation.",
        "Demonstrate that completion predicts held-out long-range periodic structure rather than local smoothness: shuffled-time, monotonic non-periodic motion, constant-speed drift, camera/pose noise, occlusion, and missing-joint controls are mandatory.",
        "Report completion correlation and spectral stability alongside downstream Period-mAP, AP50, AP75, AvgMAE, and AvgOBO; a lower pretext loss alone is not evidence.",
        "Use three independent seeds and paired source-cluster bootstrap intervals on the frozen source-disjoint development protocol.",
        "Kill the contribution if interpolation is within 2 percent of completion loss, shuffled time retains over 70 percent of Period-mAP, or the pretext fails to beat direct masked motion and PAMS under matched compute.",
        "Do not promote this candidate to the primary method unless it beats all mechanism-matched baselines and the residual can be stated without claiming any established component."
      ]
    },
    "identity-private-state-isolation": {
      "verdict": {
        "code": "REJECT_AS_METHOD_NOVELTY_REQUIRE_AS_CAUSAL_DIAGNOSTIC",
        "status": "same-family provisional",
        "confidence": "high",
        "reason": "Per-object or per-track memory and identity-preserving recurrent queries are established tracking mechanisms, while MultiCounter and MultiCounter+ already perform identity-indexed MRAC with temporal consistency. The proposed cross-person interference test is valuable, but the private-state architecture is not a defensible standalone novelty."
      },
      "overlap": [
        {
          "prior": "MultiCounter",
          "source_ids": [
            "S01"
          ],
          "collision": "Instance-specific queries, multi-person temporal modeling, tracking, and per-person counts."
        },
        {
          "prior": "MultiCounter+",
          "source_ids": [
            "S02"
          ],
          "collision": "Spatial-temporal consistency, variable periods, robust multi-person tracking/counting."
        },
        {
          "prior": "MeMOT",
          "source_ids": [
            "S18"
          ],
          "collision": "A separate long spatio-temporal identity memory for each tracked object."
        },
        {
          "prior": "TrackFormer and MOTR",
          "source_ids": [
            "S19",
            "S20"
          ],
          "collision": "Identity-preserving track queries transferred and updated over time."
        },
        {
          "prior": "Generic recurrent sequence processing",
          "source_ids": [],
          "collision": "Separate hidden state keyed by independent sequence or track ID is standard batching/state-management practice."
        }
      ],
      "residual": {
        "narrow_claim_candidate": "A repetition-counting-specific counterfactual interference diagnostic: perturb one person's tempo while holding all other tracks fixed, then measure changes in untouched identities' responses and counts.",
        "why_only_evaluation_contribution": [
          "The test directly measures functional isolation rather than assuming it from tensor indexing.",
          "It connects tracking identity state to the scientific endpoint of per-person count stability.",
          "No checked source was found to report this exact untouched-identity counterfactual for MRAC, but absence does not establish novelty."
        ],
        "permitted_role": "Required causal ablation and diagnostic for the leading method, not an independent paper method.",
        "allowed_language_if_and_only_if_supported": "We use an identity-conditioned tempo-swap intervention to quantify cross-person count interference."
      },
      "forbidden_claims": [
        "first identity-aware or identity-private multi-person counter",
        "first per-track recurrent memory",
        "first per-object temporal state",
        "first identity-preserving query or memory",
        "private buffers are novel architecture",
        "no cross-person interference merely because attention is disabled",
        "constant-time or person-count-insensitive execution without measured scaling and O(P) state disclosure",
        "end-to-end when external detector, pose estimator, tracker, or supplied tracks are used",
        "method novelty based on dictionary-keyed state allocation"
      ],
      "must_run": [
        "Use supplied tracks first so association is fixed, then separately evaluate predicted tracks with HOTA, IDF1, and ID switches.",
        "Compare identity-private state, shared scene state, stateless windows, private state with periodic resets, private state with shuffled identity keys, and private state with only track length/confidence inputs.",
        "Run single-person tempo-swap interventions while all other pose trajectories and timestamps are byte-identical; report per-frame response change, routing-history divergence, continuous count-mass change, and rounded count change for untouched identities.",
        "Include simultaneous interventions, equal-tempo people, identity switches, fragmentation, re-entry, missing frames, and state-cache eviction.",
        "Verify by instrumentation that recurrent state, routing history, overlap-add buffers, caches, and decoder accumulators never cross track IDs.",
        "Measure whether private state merely encodes track length or confidence with probing and reset controls.",
        "Report three independent seeds, paired source-cluster bootstrap intervals, and the predeclared untouched-identity thresholds.",
        "Profile latency and memory versus person count and disclose O(P) identity state; do not translate shared parameters into constant-time claims.",
        "Treat failure of the private-state benefit as a mechanism rejection, while retaining the diagnostic in the paper if it exposes cross-person coupling."
      ]
    }
  },
  "synthesis": {
    "ranked_disposition": [
      {
        "rank": 1,
        "candidate": "warp-equivariant-private-phase-flow",
        "role": "Only primary bounded pilot",
        "disposition": "Advance to a kill-oriented development pilot, with no novelty-cleared manuscript claim."
      },
      {
        "rank": 2,
        "candidate": "masked-geometric-tssm-ridge-completion",
        "role": "Mechanism-matched control or fallback pretext",
        "disposition": "Implement only if useful for the comparison matrix; do not present as primary novelty."
      },
      {
        "rank": 3,
        "candidate": "identity-private-state-isolation",
        "role": "Mandatory causal diagnostic",
        "disposition": "Integrate as an ablation/intervention around the leading method; reject as standalone architecture novelty."
      }
    ],
    "defensible_future_claim_boundary": "If eligible experiments support it, the narrow contribution may be framed as studying an explicit local time-warp transformation law for identity-indexed phase increments in pose-driven MRAC, with a counting objective that uses no per-person count or period labels. The paper must attribute phase learning, temporal equivariance, skeleton SSL, TSSM, local tempo stress, tracking memory, response decoding, and overlap-add to prior work.",
    "non_contributions": [
      "MRAC or per-person output",
      "Asynchronous or variable-speed counting",
      "Pose or skeleton input",
      "Self-supervised periodic representation",
      "Temporal self-similarity",
      "Masked pose or motion reconstruction",
      "Phase-sensitive representation",
      "Temporal transformation equivariance",
      "Local windows or localized speed perturbation",
      "Per-track memory",
      "Multi-expert terminology",
      "Normalized overlap-add",
      "One final response decode"
    ],
    "minimum_paper_story_if_successful": [
      "Question: does an explicit warp transformation law improve identity-local phase tracking beyond PAMS, SimPer, CycleCL, DeepPhase-style phase learning, and ordinary time-equivariant contrast?",
      "Mechanism: signed local phase increments governed by known augmentation derivatives inside identity-private state.",
      "Causal evidence: derivative-law ablation and identity tempo-swap isolation.",
      "Stress evidence: stronger predeclared within-track tempo profiles than the PAMS supplement, reported as an extension rather than a first.",
      "Protocol evidence: leakage-free MultiRep split, supplied and predicted tracks separated, three seeds, cluster bootstrap, period and count metrics."
    ],
    "kill_argument": "If the explicit derivative-law loss does not outperform matched augmentation-only, SimPer/CycleCL, DeepPhase-style, PAMS/Track-PAMS, and time-equivariant baselines under within-track tempo drift, there is no residual method contribution. Private state and masked TSSM remain controls, not substitutes for a failed core."
  },
  "caveats": [
    "This is a same-family GPT-5.6-Sol review and is therefore provisional, not an independent acceptance decision.",
    "The search is broad and primary-source constrained but cannot prove universal absence; no 'first' claim is authorized.",
    "TWCRAC's SSRN abstract and primary record were accessible, but reliable full-text method extraction was not. Its exact overlap is unresolved and must be closed before claim freeze.",
    "The PAMS main paper and supplement are 2026 publications and materially change the boundary: PAMS already includes localized internal tempo fluctuation experiments.",
    "Search snippets were used only to discover primary sources; verdict statements are bound to the primary source ledger.",
    "The server audit reports zero eligible method results. v46, v62, and v63 provide no novelty or efficacy evidence for any candidate.",
    "No canonical implementation exists according to the bounded server inventory. Proposed mechanisms are hypotheses, not implemented facts.",
    "External detector, pose-estimator, tracker, synthetic warp correspondence, and any video-level labels remain supervision and must be disclosed even when per-person count and period labels are excluded from the counting objective.",
    "ICASSP's short paper format increases the burden to select one narrow mechanism and one decisive diagnostic rather than combining all three candidates."
  ],
  "overall": {
    "verdict": "PROVISIONAL_ADVANCE_ONE_KILL_ORIENTED_PILOT_NO_NOVELTY_CLEARANCE",
    "decision": "Advance warp-equivariant-private-phase-flow as the only bounded primary pilot; retain masked-geometric-tssm-ridge-completion as a control and identity-private-state-isolation as a causal diagnostic. Do not freeze a paper method or novelty claim.",
    "actions": [
      {
        "priority": 1,
        "action": "Correct the literature landscape to include PAMS's localized internal tempo perturbation and DeepPhase's unsupervised local motion-phase manifold."
      },
      {
        "priority": 2,
        "action": "Freeze a leakage-free, source-disjoint MultiRep development protocol, checksums, evaluator, supervision contract, and three-seed statistical plan before model selection."
      },
      {
        "priority": 3,
        "action": "Implement the smallest derivative-law phase-flow pilot with matched SimPer, CycleCL, DeepPhase-style, time-equivariant, PAMS, and Track-PAMS baselines."
      },
      {
        "priority": 4,
        "action": "Run the identity tempo-swap intervention and the derivative-law kill ablation before adding experts, overlap-add variants, or auxiliary losses."
      },
      {
        "priority": 5,
        "action": "Use masked geometric TSSM completion only as a matched pretext control unless it independently clears the stronger prior matrix."
      },
      {
        "priority": 6,
        "action": "Obtain and inspect the complete TWCRAC paper, then rerun this novelty boundary before manuscript claim freeze."
      },
      {
        "priority": 7,
        "action": "Keep all manuscript claims provisional and numerical result fields empty until experiment-audit and result-to-claim gates pass."
      }
    ],
    "paper_status": "blocked_for_claims",
    "eligible_method_results": 0,
    "review_complete": true
  }
}

