
from analyzer import (
    load_transactions,
    analyze_transactions,
    prepare_data,
    train_model,
    evaluate_model,
    predict_transaction,
    export_results
)


def main():
    print("=" * 50)
    print("       CREDIT CARD FRAUD DETECTION")
    print("=" * 50)

    try:
        df = load_transactions("transactions.csv")

        analyze_transactions(df)

        X_train, X_test, y_train, y_test = prepare_data(df)

        model = train_model(X_train, y_train)

        predictions = evaluate_model(model, X_test, y_test)

        predict_transaction(
            model,
            amount=25000,
            hour=2,
            international=1,
            online=1,
            previous_fraud=1,
            transactions_last_24h=15
        )

        export_results(X_test, y_test, predictions)

        print("\nProject completed successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
