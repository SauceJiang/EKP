# Lightweight Multimodal Sequence Regression Framework for Enzyme Kinetic Parameter Prediction

```mermaid
flowchart TB
    protein_input[Protein Sequence]
    compound_input[Compound Sequence]

    protein_enc[Protein Encoder]
    compound_enc[Compound Encoder]

    protein_pool[Pooling Module]
    compound_pool[Pooling Module]

    fusion[Fusion Module]
    reg_head[Regression Head]
    output[Predict log10(kcat)]

    protein_input --> protein_enc --> protein_pool --> fusion
    compound_input --> compound_enc --> compound_pool --> fusion
    fusion --> reg_head --> output
```
