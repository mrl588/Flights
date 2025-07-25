#PythonNotebook
#%%

import pandas as pd


dtypes = {
    "F_DEPARTURE_CITY": "string",  
    "F_ARRIVAL_CITY": "string",
    "F_TRAVELER_COUNT": "int16",   
    "P_ORDER_TOTAL_AMOUNT": "float32",
    "F_CABIN_CLASS": "string",
    "F_USER_ID": "string",        
  
}

# Define date columns for efficient parsing
date_columns = ["F_DEPARTURE_TIME", "F_EMAIL_TIME"]

df = pd.read_csv(
    "flights.csv", 
    usecols=[
        "F_DEPARTURE_CITY", 
        "F_ARRIVAL_CITY", 
        "F_DEPARTURE_TIME", 
        "F_TRAVELER_COUNT", 
        "P_ORDER_TOTAL_AMOUNT", 
        "F_CABIN_CLASS", 
        "F_USER_ID", 
        'F_EMAIL_TIME',
        "F_MERCHANT_NAME"
    ],
    dtype=dtypes,
    parse_dates=date_columns,  # Parse dates during load
    infer_datetime_format=True,  # Speed up date parsing
    low_memory=False,  # Prevent column type inference errors
    na_values=['', 'NA', 'NULL', 'NaN'],  # Standardize NA values
    cache_dates=True  # Cache dates for faster parsing
)

df = df.dropna(subset=["F_DEPARTURE_TIME", "F_TRAVELER_COUNT", "P_ORDER_TOTAL_AMOUNT", "F_CABIN_CLASS", "F_USER_ID", "F_EMAIL_TIME", "F_MERCHANT_NAME", "F_DEPARTURE_CITY", "F_ARRIVAL_CITY"])


# %%
df["FLIGHT_LEG"] = df["F_DEPARTURE_CITY"] + " -- " + df["F_ARRIVAL_CITY"]
top_10_legs = df["FLIGHT_LEG"].value_counts().head(10)
print(top_10_legs)
# The top 10 flight legs are printed to the console

# %%
top_10_legs = df["FLIGHT_LEG"].value_counts().head(10).index.tolist()
filtered_df = df[df["FLIGHT_LEG"].isin(top_10_legs)] # Filter the DataFrame to include only the top 10 legs good for future analysis
filtered_df.loc[:, "PRICE_PER_TRAVELER"] = filtered_df["P_ORDER_TOTAL_AMOUNT"] / filtered_df["F_TRAVELER_COUNT"]
medmean = filtered_df.groupby("FLIGHT_LEG")["PRICE_PER_TRAVELER"].agg(["mean", "median"]).reset_index()
print(medmean) 

# %%

filtered_df = df[df["FLIGHT_LEG"].isin(top_10_legs)].copy()

filtered_df = filtered_df[(filtered_df["P_ORDER_TOTAL_AMOUNT"] > 0) & 
                         (filtered_df["F_TRAVELER_COUNT"] > 0)]

agg = filtered_df.groupby(["FLIGHT_LEG", "F_MERCHANT_NAME", "F_CABIN_CLASS"]).agg(
    total_price=("P_ORDER_TOTAL_AMOUNT", "sum"),
    cabin_total_travelers=("F_TRAVELER_COUNT", "sum")
).reset_index()


agg["avg_price_per_traveler"] = agg["total_price"] / agg["cabin_total_travelers"]
min_reasonable_price = 10  
agg = agg[agg["avg_price_per_traveler"] >= min_reasonable_price]

total_travelers_per_leg = agg.groupby("FLIGHT_LEG")["cabin_total_travelers"].transform("sum")
agg["weight"] = agg["cabin_total_travelers"] / total_travelers_per_leg


weighted_avg = agg.groupby(["FLIGHT_LEG", "F_MERCHANT_NAME"]).apply(
    lambda x: (x["avg_price_per_traveler"] * x["weight"]).sum()
).reset_index(name="weighted_avg_price")

weighted_avg = weighted_avg.sort_values(by=["FLIGHT_LEG", "weighted_avg_price"])


cheapest_airlines = weighted_avg.loc[weighted_avg.groupby("FLIGHT_LEG")["weighted_avg_price"].idxmin()]
cheapest_airlines = cheapest_airlines.rename(columns={
    "F_MERCHANT_NAME": "Cheapest Airline", 
    "weighted_avg_price": "Lowest Price"
})


expensive_airlines = weighted_avg.loc[weighted_avg.groupby("FLIGHT_LEG")["weighted_avg_price"].idxmax()]
expensive_airlines = expensive_airlines.rename(columns={
    "F_MERCHANT_NAME": "Most Expensive Airline", 
    "weighted_avg_price": "Highest Price"
})

cheapest = cheapest_airlines[["FLIGHT_LEG", "Cheapest Airline", "Lowest Price"]]
expensive = expensive_airlines[["FLIGHT_LEG", "Most Expensive Airline", "Highest Price"]]
result = pd.merge(cheapest, expensive, on="FLIGHT_LEG")


result["Lowest Price"] = result["Lowest Price"].round(2)
result["Highest Price"] = result["Highest Price"].round(2)


print("Most Expensive and Cheapest Airlines for Top 10 Flight Legs:")
print(result.sort_values("FLIGHT_LEG"))

# %%
 # allows for date time conversion or to filter based on time
df["F_DEPARTURE_TIME"] = pd.to_datetime(df["F_DEPARTURE_TIME"], errors="coerce")
mask = (
    (df["F_DEPARTURE_TIME"].dt.year == 2023) &
    (
        ((df["F_DEPARTURE_CITY"] == "New York") & (df["F_ARRIVAL_CITY"] == "Chicago")) |
        ((df["F_DEPARTURE_CITY"] == "Chicago") & (df["F_ARRIVAL_CITY"] == "New York"))
    )
)
filtered_df = df[mask]

counts_per_user = filtered_df.groupby("F_USER_ID").size()

average_flight_frequency = counts_per_user.mean()
print(f"Average flight frequency for users flying between New York and Chicago in 2023: {average_flight_frequency:.2f}")



# %%

df["F_EMAIL_TIME"] = pd.to_datetime(df["F_EMAIL_TIME"], errors="coerce")
df["F_DEPARTURE_TIME"] = pd.to_datetime(df["F_DEPARTURE_TIME"], errors="coerce")


df["# Days Booked Ahead"] = (df["F_DEPARTURE_TIME"] - df["F_EMAIL_TIME"]).dt.days
df = df[df["# Days Booked Ahead"] >= 0]
df = df[df["F_TRAVELER_COUNT"] >= 0]
df = df[df["P_ORDER_TOTAL_AMOUNT"] >30]
df["PRICE_PER_TRAVELER"] = df["P_ORDER_TOTAL_AMOUNT"] / df["F_TRAVELER_COUNT"]

filtered_df = df[df["FLIGHT_LEG"].isin(top_10_legs)].copy()

Q6_filter = filtered_df.groupby(["FLIGHT_LEG", "# Days Booked Ahead"])["PRICE_PER_TRAVELER"].mean().reset_index()

best_booking_time = Q6_filter.loc[Q6_filter.groupby("FLIGHT_LEG")["PRICE_PER_TRAVELER"].idxmin()]


print("Best time to book for top 10 flight legs to get lowest prices:")
print(best_booking_time[["FLIGHT_LEG", "# Days Booked Ahead", "PRICE_PER_TRAVELER"]]) 


# %%

df = df[df["F_CABIN_CLASS"].notnull()]

def normalize_cabin_class(cabin):
    cabin = str(cabin).lower()
    if "basic" in cabin or "economy" in cabin:
        return "Economy"
    elif "main" in cabin:
        return "Main Cabin"
    elif "premium" in cabin:
        return "Premium Economy"
    elif "business" in cabin or "first" in cabin:
        return "Business/First Class"
    else:
        return "Other"

df["CABIN_CLASS_GROUPED"] = df["F_CABIN_CLASS"].apply(normalize_cabin_class)

most_popular_leg = df["FLIGHT_LEG"].value_counts().idxmax()
leg_df = df[df["FLIGHT_LEG"] == most_popular_leg].copy()
leg_df["PRICE_PER_TRAVELER"] = leg_df["P_ORDER_TOTAL_AMOUNT"] / leg_df["F_TRAVELER_COUNT"]

avg_price_per_cabin = leg_df.groupby("CABIN_CLASS_GROUPED")["PRICE_PER_TRAVELER"].mean()
print("Average price per traveler by cabin class:")
print(avg_price_per_cabin)

if "Economy" in avg_price_per_cabin.index:
    economy_price = avg_price_per_cabin["Economy"]
    pct_diff = ((avg_price_per_cabin - economy_price) / economy_price) * 100
    print("\nPercentage price difference relative to Economy class:")
    print(pct_diff)
else:
    print("Economy class not found in this flight leg. Showing raw average prices instead.")


# %%
# Ensure datetime
df["F_DEPARTURE_TIME"] = pd.to_datetime(df["F_DEPARTURE_TIME"], errors="coerce")


# Price per traveler
df["PRICE_PER_TRAVELER"] = df["P_ORDER_TOTAL_AMOUNT"] / df["F_TRAVELER_COUNT"]

# Filter for NYC <-> Chicago routes
nyc_chicago_mask = (
    ((df["F_DEPARTURE_CITY"] == "New York") & (df["F_ARRIVAL_CITY"] == "Chicago")) |
    ((df["F_DEPARTURE_CITY"] == "Chicago") & (df["F_ARRIVAL_CITY"] == "New York"))
)
nyc_chicago_df = df[nyc_chicago_mask].copy()

# Extract month
nyc_chicago_df["Month"] = nyc_chicago_df["F_DEPARTURE_TIME"].dt.month

# Group by month and get average price
monthly_avg_price = nyc_chicago_df.groupby("Month")["PRICE_PER_TRAVELER"].mean().reset_index()

# Find the cheapest month
cheapest_month = monthly_avg_price.loc[monthly_avg_price["PRICE_PER_TRAVELER"].idxmin()]
print("Cheapest Month for NYC <-> Chicago:")
print(cheapest_month)


nyc_boston_df = df[df["FLIGHT_LEG"] == "New York -- Boston"].copy()
cheapest_flight = nyc_boston_df.loc[nyc_boston_df["PRICE_PER_TRAVELER"].idxmin()]

print(f"Departure Date: {cheapest_flight['F_DEPARTURE_TIME'].date()}")
print(f" Price: {cheapest_flight['P_ORDER_TOTAL_AMOUNT']}")


