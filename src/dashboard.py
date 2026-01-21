import matplotlib.pyplot as plt

def supplier_dashboard(df):
    df.sort_values("risk_score").plot(
        x="Suppliers", y="Risk Score", kind="bar"
    )
    plt.title("Supplier Risk Scores")
    plt.tight_layout()
    plt.show()
