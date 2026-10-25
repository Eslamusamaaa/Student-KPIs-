"""Run the whole pipeline: augmentation -> EDA figures -> training and evaluation.

    python run_pipeline.py
"""
from src import data_augmentation, eda, train

if __name__ == "__main__":
    print("== 1/3 Augmenting the real student records ==")
    data_augmentation.main()
    print("\n== 2/3 Exploratory analysis ==")
    eda.main()
    print("\n== 3/3 Training and evaluation ==")
    train.main()
