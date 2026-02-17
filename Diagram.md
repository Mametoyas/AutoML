# AutoML System Architecture Diagrams

## 1. AutoML System Overview

```mermaid
graph TB
    Start([Start]) --> Input[Input Dataset]
    Input --> TaskType{Task Type?}
    TaskType -->|Classification| ClassTask[Classification Task]
    TaskType -->|Regression| RegTask[Regression Task]
    
    ClassTask --> AutoML[AutoML Optimizer]
    RegTask --> AutoML
    
    AutoML --> MetaOpt[Metaheuristic Optimization]
    MetaOpt --> SearchSpace[Search Space Exploration]
    
    SearchSpace --> Preprocess[Preprocessing Selection]
    SearchSpace --> ModelSel[Model Selection]
    SearchSpace --> HyperParam[Hyperparameter Tuning]
    
    Preprocess --> Evaluate[Fitness Evaluation]
    ModelSel --> Evaluate
    HyperParam --> Evaluate
    
    Evaluate --> Constraint{Constraints<br/>Satisfied?}
    Constraint -->|No| MetaOpt
    Constraint -->|Yes| BestConfig[Best Configuration]
    
    BestConfig --> Train[Train Final Model]
    Train --> Output[Optimized Model]
    Output --> End([End])
    
    style AutoML fill:#ff9999
    style MetaOpt fill:#99ccff
    style Evaluate fill:#99ff99
    style BestConfig fill:#ffcc99
```

## 2. Search Space Architecture

```mermaid
graph LR
    SearchSpace[Search Space] --> Dim1[Dimension 1:<br/>Preprocessing]
    SearchSpace --> Dim2[Dimension 2:<br/>Model Selection]
    SearchSpace --> Dim3[Dimension 3:<br/>Hyperparameters]
    
    Dim1 --> Scale[Feature Scaling]
    Dim1 --> Select[Feature Selection]
    
    Scale --> S1[None]
    Scale --> S2[Standard Scaling]
    Scale --> S3[Min-Max Scaling]
    
    Select --> F1[None]
    Select --> F2[SelectKBest]
    Select --> F3[PCA]
    
    Dim2 --> M1[Logistic Regression]
    Dim2 --> M2[SVM]
    Dim2 --> M3[Random Forest]
    Dim2 --> M4[XGBoost]
    Dim2 --> M5[Neural Network]
    
    Dim3 --> H1[Learning Rate]
    Dim3 --> H2[N Estimators]
    Dim3 --> H3[Max Depth]
    Dim3 --> H4[Min Samples Split]
    Dim3 --> H5[Regularization]
    
    style SearchSpace fill:#ff99cc
    style Dim1 fill:#99ccff
    style Dim2 fill:#99ff99
    style Dim3 fill:#ffcc99
```

## 3. Metaheuristic Optimization Flow

```mermaid
graph TB
    Init[Initialize Population] --> Eval1[Evaluate Fitness]
    Eval1 --> Gen{Max Generation<br/>Reached?}
    
    Gen -->|No| Select[Selection]
    Select --> Crossover[Crossover]
    Crossover --> Mutation[Mutation]
    Mutation --> NewPop[New Population]
    
    NewPop --> Eval2[Evaluate Fitness]
    Eval2 --> CheckConst{Check<br/>Constraints}
    
    CheckConst -->|Violated| Penalty[Apply Penalty]
    CheckConst -->|Satisfied| UpdateBest[Update Best Solution]
    
    Penalty --> Gen
    UpdateBest --> Gen
    
    Gen -->|Yes| BestSol[Return Best Solution]
    BestSol --> End([Optimized Configuration])
    
    style Init fill:#99ccff
    style Eval1 fill:#99ff99
    style Eval2 fill:#99ff99
    style BestSol fill:#ffcc99
```

## 4. Preprocessing Pipeline

```mermaid
graph LR
    RawData[Raw Dataset] --> Missing[Handle Missing Values]
    Missing --> Encode[Encode Categorical]
    Encode --> Split[Train-Test Split]
    
    Split --> ScaleChoice{Scaling<br/>Method?}
    
    ScaleChoice -->|None| NoScale[No Scaling]
    ScaleChoice -->|Standard| StdScale[Standard Scaler]
    ScaleChoice -->|MinMax| MMScale[MinMax Scaler]
    
    NoScale --> FeatureChoice{Feature<br/>Selection?}
    StdScale --> FeatureChoice
    MMScale --> FeatureChoice
    
    FeatureChoice -->|None| NoSelect[All Features]
    FeatureChoice -->|SelectKBest| KBest[SelectKBest]
    FeatureChoice -->|PCA| PCA[PCA Transform]
    
    NoSelect --> Ready[Ready for Training]
    KBest --> Ready
    PCA --> Ready
    
    style RawData fill:#ff99cc
    style Ready fill:#99ff99
```

## 5. Model Selection & Training

```mermaid
graph TB
    PrepData[Preprocessed Data] --> ModelChoice{Select Model}
    
    ModelChoice -->|Option 1| LR[Logistic Regression]
    ModelChoice -->|Option 2| SVM[Support Vector Machine]
    ModelChoice -->|Option 3| RF[Random Forest]
    ModelChoice -->|Option 4| XGB[XGBoost]
    ModelChoice -->|Option 5| NN[Neural Network]
    
    LR --> LR_HP[HP: C, penalty, solver]
    SVM --> SVM_HP[HP: C, kernel, gamma]
    RF --> RF_HP[HP: n_estimators, max_depth]
    XGB --> XGB_HP[HP: learning_rate, n_estimators]
    NN --> NN_HP[HP: layers, neurons, activation]
    
    LR_HP --> Train[Train Model]
    SVM_HP --> Train
    RF_HP --> Train
    XGB_HP --> Train
    NN_HP --> Train
    
    Train --> Validate[Cross-Validation]
    Validate --> Metrics[Calculate Metrics]
    
    style PrepData fill:#99ccff
    style Train fill:#ff99cc
    style Metrics fill:#99ff99
```

## 6. Fitness Evaluation Process

```mermaid
graph TB
    Model[Trained Model] --> TaskCheck{Task Type?}
    
    TaskCheck -->|Classification| ClassMetrics[Classification Metrics]
    TaskCheck -->|Regression| RegMetrics[Regression Metrics]
    
    ClassMetrics --> Acc[Accuracy]
    ClassMetrics --> F1[F1-Score]
    ClassMetrics --> ROC[ROC-AUC]
    
    RegMetrics --> RMSE[RMSE]
    RegMetrics --> MAE[MAE]
    RegMetrics --> R2[R² Score]
    
    Acc --> Combine[Combine Metrics]
    F1 --> Combine
    ROC --> Combine
    RMSE --> Combine
    MAE --> Combine
    R2 --> Combine
    
    Combine --> Fitness[Fitness Score]
    
    Fitness --> ConstCheck{Check<br/>Constraints}
    
    ConstCheck --> Time{Time Limit<br/>OK?}
    ConstCheck --> Complex{Complexity<br/>OK?}
    ConstCheck --> Memory{Memory<br/>OK?}
    
    Time -->|No| Penalize[Apply Penalty]
    Complex -->|No| Penalize
    Memory -->|No| Penalize
    
    Time -->|Yes| Final[Final Fitness]
    Complex -->|Yes| Final
    Memory -->|Yes| Final
    
    Penalize --> Final
    
    style Fitness fill:#99ff99
    style Final fill:#ffcc99
```

## 7. Complete AutoML Workflow

```mermaid
graph TB
    subgraph Input
        Data[Dataset] --> DataInfo[Data Analysis]
        Task[Task Definition] --> TaskInfo[Classification/Regression]
    end
    
    subgraph Initialization
        DataInfo --> InitPop[Initialize Population]
        TaskInfo --> InitPop
        InitPop --> Config1[Config 1]
        InitPop --> Config2[Config 2]
        InitPop --> Config3[Config N]
    end
    
    subgraph Optimization Loop
        Config1 --> Pipeline1[Build Pipeline 1]
        Config2 --> Pipeline2[Build Pipeline 2]
        Config3 --> Pipeline3[Build Pipeline N]
        
        Pipeline1 --> Eval1[Evaluate 1]
        Pipeline2 --> Eval2[Evaluate 2]
        Pipeline3 --> Eval3[Evaluate N]
        
        Eval1 --> Fitness1[Fitness 1]
        Eval2 --> Fitness2[Fitness 2]
        Eval3 --> Fitness3[Fitness N]
        
        Fitness1 --> MetaAlg[Metaheuristic Algorithm]
        Fitness2 --> MetaAlg
        Fitness3 --> MetaAlg
        
        MetaAlg --> NextGen{Next<br/>Generation?}
        NextGen -->|Yes| Config1
        NextGen -->|No| BestConfig[Best Configuration]
    end
    
    subgraph Output
        BestConfig --> FinalTrain[Train Final Model]
        FinalTrain --> FinalModel[Optimized Model]
        FinalModel --> Deploy[Ready for Deployment]
    end
    
    style Data fill:#ff99cc
    style MetaAlg fill:#99ccff
    style FinalModel fill:#99ff99
    style Deploy fill:#ffcc99
```

## 8. Constraint Handling System

```mermaid
graph LR
    Solution[Candidate Solution] --> C1{Training Time}
    
    C1 -->|> Limit| Reject1[Penalty/Reject]
    C1 -->|<= Limit| C2{Model Complexity}
    
    C2 -->|Too Complex| Reject2[Penalty/Reject]
    C2 -->|Acceptable| C3{Memory Usage}
    
    C3 -->|> Limit| Reject3[Penalty/Reject]
    C3 -->|<= Limit| Valid[Valid Solution]
    
    Reject1 --> Adjust[Adjust Fitness]
    Reject2 --> Adjust
    Reject3 --> Adjust
    
    Adjust --> Return[Return to Pool]
    Valid --> Accept[Accept Solution]
    
    style Solution fill:#99ccff
    style Valid fill:#99ff99
    style Accept fill:#ffcc99
```

## 9. Hyperparameter Space per Model

```mermaid
graph TB
    subgraph Logistic Regression
        LR[Logistic Regression] --> LR1[C: 0.001-100]
        LR --> LR2[penalty: l1, l2, elasticnet]
        LR --> LR3[solver: liblinear, saga]
    end
    
    subgraph SVM
        SVM[Support Vector Machine] --> SVM1[C: 0.1-100]
        SVM --> SVM2[kernel: linear, rbf, poly]
        SVM --> SVM3[gamma: scale, auto, 0.001-1]
    end
    
    subgraph Random Forest
        RF[Random Forest] --> RF1[n_estimators: 10-500]
        RF --> RF2[max_depth: 3-50]
        RF --> RF3[min_samples_split: 2-20]
        RF --> RF4[min_samples_leaf: 1-10]
    end
    
    subgraph XGBoost
        XGB[XGBoost] --> XGB1[learning_rate: 0.01-0.3]
        XGB --> XGB2[n_estimators: 50-500]
        XGB --> XGB3[max_depth: 3-15]
        XGB --> XGB4[subsample: 0.5-1.0]
    end
    
    subgraph Neural Network
        NN[Neural Network] --> NN1[hidden_layers: 1-5]
        NN --> NN2[neurons: 16-512]
        NN --> NN3[activation: relu, tanh, sigmoid]
        NN --> NN4[learning_rate: 0.0001-0.1]
        NN --> NN5[dropout: 0.0-0.5]
    end
    
    style LR fill:#ff9999
    style SVM fill:#99ccff
    style RF fill:#99ff99
    style XGB fill:#ffcc99
    style NN fill:#ff99cc
```

## 10. Performance Evaluation Flow

```mermaid
graph TB
    Start([Start Evaluation]) --> LoadData[Load Preprocessed Data]
    LoadData --> CV[K-Fold Cross-Validation]
    
    CV --> Fold1[Fold 1]
    CV --> Fold2[Fold 2]
    CV --> FoldK[Fold K]
    
    Fold1 --> Train1[Train on K-1 Folds]
    Fold2 --> Train2[Train on K-1 Folds]
    FoldK --> TrainK[Train on K-1 Folds]
    
    Train1 --> Test1[Test on 1 Fold]
    Train2 --> Test2[Test on 1 Fold]
    TrainK --> TestK[Test on 1 Fold]
    
    Test1 --> Score1[Score 1]
    Test2 --> Score2[Score 2]
    TestK --> ScoreK[Score K]
    
    Score1 --> Aggregate[Aggregate Scores]
    Score2 --> Aggregate
    ScoreK --> Aggregate
    
    Aggregate --> Mean[Mean Score]
    Aggregate --> Std[Std Deviation]
    
    Mean --> Final[Final Fitness Score]
    Std --> Final
    
    Final --> End([Return Score])
    
    style CV fill:#99ccff
    style Aggregate fill:#99ff99
    style Final fill:#ffcc99
```

---

## Legend

- 🔵 **Blue**: Initialization/Input stages
- 🟢 **Green**: Evaluation/Validation stages
- 🟡 **Orange**: Output/Result stages
- 🔴 **Red**: Core AutoML components
- 🟣 **Purple**: Data processing stages
