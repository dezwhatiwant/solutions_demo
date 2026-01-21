import matplotlib.pyplot as plt

def supplier_dashboard(df):
    df.sort_values("absolute_risk").plot(
        x="supplier_clean", y="absolute_risk", kind="bar"
    )
    plt.title("Supplier Risk Assesment")
    plt.xlabel("Suppliers")
    plt.ylim(0.35, 0.85)
    plt.tight_layout()
    plt.show()
