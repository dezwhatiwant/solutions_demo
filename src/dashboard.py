import matplotlib.pyplot as plt

def supplier_dashboard(df):
    df.sort_values("risk_score").plot(
        x="supplier_clean", y="risk_score", kind="bar"
    )
    plt.title("Supplier Risk Scores")
    plt.tight_layout()
    plt.show()
