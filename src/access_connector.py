import pandas as pd
import pyodbc

def load_access_data(db_path):
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        rf'DBQ={db_path};'
    )

    try:
        conn = pyodbc.connect(conn_str)
        print("✅ Connected to database")

        # Load individual tables
        tickets_df = pd.read_sql("SELECT * FROM Tickets", conn)
        priority_df = pd.read_sql("SELECT * FROM Ticket_Priority", conn)
        department_df = pd.read_sql("SELECT * FROM Department", conn)

        print(f"Tickets rows: {len(tickets_df)}")
        print(f"Priority rows: {len(priority_df)}")
        print(f"Departments rows: {len(department_df)}")

        print("📌 Sample ticket data:\n", tickets_df.head())

        # Merge step-by-step
        df = pd.merge(tickets_df, priority_df, on='ticket_id', how='inner')
        print("✅ Merged with priority")

        df = pd.merge(df, department_df, on='department_id', how='inner')
        print("✅ Merged with department")

        conn.close()
        return df

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

if __name__ == "__main__":
    db_path = r"D:\Users\balans\Desktop\PBI\Customer Support Ticket Prioritization\data\customer_tickets.accdb"
    df = load_access_data(db_path)

    if df is not None:
        print("\n📊 Final merged dataframe:\n")
        print(df.head())

        # Save merged dataframe to CSV
        df.to_csv("D:/Users/balans/Desktop/PBI/Customer Support Ticket Prioritization/data/merged_tickets.csv", index=False)
        print("\n\n💾 Saved merged data to 'data/merged_tickets.csv'")
    else:
        print("⚠️ No data returned.")
