import matplotlib.pyplot as plt

def supplier_dashboard(df):
    df.sort_values("score").plot(
        x="supplier_clean", y="score", kind="bar"
    )
    plt.title("Supplier Risk Assesment")
    plt.xlabel("Suppliers")
    plt.ylim(0.35, 0.75)
    plt.tight_layout()
    plt.show()
