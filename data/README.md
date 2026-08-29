# Dataset placement

Copy or link the original CG-LSMN Method--Class dataset to this layout:

```text
data/
└── Dataset_Method_Class/
    ├── new_labels.txt
    ├── vocabDict.json
    └── allDataDict.json
```

The frozen paper dataset contains 20,371 graph pairs. Expected SHA-256 values:

```text
new_labels.txt  0f8f935359c0c625c953b9e4d168293dbb1e529983d73654435fe3babb6b642c
vocabDict.json  ad7883163ecc6dc767b199a6c15d0c777db70d8c5bb72ee5294072aff0fee335
allDataDict.json 6832cf920a7f2e80f63a03d528dbd46c3f3a410e2fd2f98c7d852a5dc7b9f8d9
```

An external location can be selected with `CGLSMN_DATA_ROOT`.
