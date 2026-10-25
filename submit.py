import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")      
os.makedirs("plots", exist_ok=True)   
df = pd.read_csv("SolarPrediction.csv")
print("Rows and columns:", df.shape)
print(df.head())   
print(df.describe())
date_only = df["Data"].str.split().str[0]    
fmt = "%m/%d/%Y %H:%M:%S"
df["DateTime"] = pd.to_datetime(date_only + " " + df["Time"], format=fmt)
df["SunRise"] = pd.to_datetime(date_only + " " + df["TimeSunRise"], format=fmt)
df["SunSet"] = pd.to_datetime(date_only + " " + df["TimeSunSet"], format=fmt)
df = df.sort_values("DateTime").reset_index(drop=True)
df["Date"] = df["DateTime"].dt.date      
df["Hour"] = df["DateTime"].dt.hour   
df["Month"] = df["DateTime"].dt.month_name()
df["IsDaylight"] = (df["DateTime"] >= df["SunRise"]) & (df["DateTime"] <= df["SunSet"])
df = df.rename(columns={"WindDirection(Degrees)": "WindDirection"})
print("Data runs from", df["DateTime"].min(), "to", df["DateTime"].max())
print(df[["Data", "Time", "DateTime", "SunRise", "SunSet", "Radiation"]].head())
daily_avg = df.groupby("Date")["Radiation"].mean()
plt.figure(figsize=(14, 5))
plt.plot(df["DateTime"], df["Radiation"], linewidth=0.3, alpha=0.6, label="Every reading")
plt.plot(pd.to_datetime(daily_avg.index), daily_avg.values, color="red", linewidth=2,label="Daily average")
plt.title("Solar radiation over the full period")
plt.xlabel("Date")
plt.ylabel("Radiation (W/m²)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/A1_radiation_timeline.png")
plt.show()
first_d = df[df["DateTime"] < "2016-09-06"]
plt.figure(figsize=(14, 4))
plt.plot(first_d["DateTime"], first_d["Radiation"], color="orange")
plt.title("Radiation over the first 5 days")
plt.xlabel("Date and time")
plt.ylabel("Radiation (W/m²)")
plt.tight_layout()
plt.show()
h_a = df.groupby("Hour")["Radiation"].mean()
plt.figure(figsize=(10, 5))
plt.plot(h_a.index, h_a.values, marker="o", color="darkorange")
plt.title("Average radiation by hour of day")
plt.xlabel("Hour of day")
plt.ylabel("Average radiation (W/m^2)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/A1_hourly_average.png")
plt.show()
print("Hour with highest average radiation:", h_a.idxmax())
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Hour", y="Radiation", hue="Month")
plt.title("Average radiation by hour, for each month")
plt.xlabel("Hour of day")
plt.ylabel("Average radiation (W/m^2)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/A1_hourly_by_month.png")
plt.show()
sample = df.sample(5000, random_state=42)
parameters = ["Temperature", "Humidity", "Pressure", "Speed"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9)) 
for ax, col in zip(axes.flat, parameters):
    sns.regplot(data=sample, x=col, y="Radiation", ax=ax,
                scatter_kws={"s": 5, "alpha": 0.3}, line_kws={"color": "red"})
    r = df[col].corr(df["Radiation"])
    ax.set_title(f"Radiation vs {col}")
plt.tight_layout()
plt.savefig("plots/A2_scatter_plots.png")
plt.show()
columns = ["Radiation", "Temperature", "Pressure", "Humidity", "WindDirection", "Speed"]
correlation = df[columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between all parameters")
plt.tight_layout()
plt.savefig("plots/A2_correlation_heatmap.png")
plt.show()
print("Correlation of each parameter with Radiation:")
print(correlation["Radiation"].drop("Radiation").sort_values(ascending=False).round(3))
plt.figure(figsize=(14, 5))
sns.boxplot(data=df, x="Hour", y="Radiation", color="orange")
plt.title("Radiation at each hour")
plt.xlabel("Hour of day")
plt.ylabel("Radiation (W/m²)")
plt.tight_layout()
plt.savefig("plots/A3_boxplot_by_hour.png")
plt.show()
cols = ["Radiation", "Temperature", "Humidity", "Pressure", "Speed"]
standardised = (df[cols] - df[cols].mean()) / df[cols].std()
plt.figure(figsize=(10, 5))
sns.boxplot(data=standardised, fliersize=2)
plt.title("All parameters on the same scale (dots = outliers)")
plt.ylabel("Standard deviations from the mean")
plt.tight_layout()
plt.savefig("plots/A3_boxplot_all_parameters.png")
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df["Radiation"], bins=60, ax=axes[0])
axes[0].set_title("All readings")
axes[0].set_xlabel("Radiation (W/m²)")
daytime = df[(df["Hour"] >= 6) & (df["Hour"] < 18)]
sns.histplot(daytime["Radiation"], bins=60, ax=axes[1], color="orange")
axes[1].set_title("Daytime readings only (6am - 6pm)")
axes[1].set_xlabel("Radiation (W/m²)")
plt.tight_layout()
plt.savefig("plots/A3_histograms.png")
plt.show()
for col in cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
print("\nThe 5 highest radiation readings:")
print(df.nlargest(5, "Radiation")[["DateTime", "Radiation", "Temperature", "Humidity"]])
h_a = df.groupby("Hour")["Radiation"].mean()
peak_hour = h_a.idxmax()         
peak_value = h_a.max()
print("Average radiation for each hour:")
print(h_a.round(1))
print("Top 3 hours:")
print(h_a.sort_values(ascending=False).head(3).round(1))
colours = ["#23ebe1" if h == peak_hour else "#eb3e0e" for h in h_a.index]
plt.figure(figsize=(11, 5))
plt.bar(h_a.index,h_a.values,color=colours)
plt.title("Average radiation by hour (red = highest)")
plt.xlabel("Hour of day")
plt.ylabel("Average radiation (W/m²)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/B1_peak_hour.png")
plt.show()
parameters = ["Temperature", "Pressure", "Humidity", "WindDirection", "Speed"]
corr_all = df[parameters + ["Radiation"]].corr()["Radiation"].drop("Radiation")
daytime = df[df["IsDaylight"]]
corr_day = daytime[parameters + ["Radiation"]].corr()["Radiation"].drop("Radiation")
corr_table = pd.DataFrame({"All readings": corr_all, "Daytime only": corr_day})
corr_table = corr_table.loc[corr_all.abs().sort_values(ascending=False).index]
print(corr_table.round(3))
corr_table.plot(kind="barh", figsize=(11, 5), color=["#1d4ade", "#f20511"])
plt.axvline(0, color="black", linewidth=1)
plt.title("Correlation of each parameter with radiation")
plt.xlabel("Correlation (r)")
plt.tight_layout()
plt.savefig("plots/B2_correlations.png")
plt.show()
strongest = corr_table.index[0]
print(f"\nANSWER B2: {strongest} has the strongest relationship with radiation "
      f"(r = {corr_all[strongest]:.2f}). The sun heats the air, so more radiation means "
      f"higher temperature.")
print(df[["Radiation", "Temperature", "Humidity"]].corr().round(2))
band_table = df.groupby("TempBand", observed=True)[["Radiation", "Humidity"]].mean().round(1)
print("\nAverage radiation and humidity in each temperature band:")
print(band_table)
sample = df.sample(5000, random_state=42)  
plt.figure(figsize=(11, 6))
points = plt.scatter(sample["Temperature"], sample["Humidity"], c=sample["Radiation"],cmap="inferno", s=8, alpha=0.7)
plt.title("Temperature vs humidity, coloured by radiation")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.savefig("plots/B3_temp_humidity_radiation.png")
plt.show()
for col in ["Radiation", "Temperature", "Pressure", "Humidity", "Speed"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"   {col:12s}: {n}")
hour_median = df.groupby("Hour")["Radiation"].transform("median")  
dips = df[(df["Hour"] >= 10) & (df["Hour"] <= 14) & (df["Radiation"] < 0.3 * hour_median)]
print(f"\n2) Midday readings below 30% of normal: {len(dips)} readings on "
      f"{dips['Date'].nunique()} different days")
night = df[~df["IsDaylight"] & (df["Radiation"] > 10)]
print(f"\n3) Readings above 10 W/m² after sunset / before sunrise: {len(night)}")
print("\n4) The 5 highest radiation readings:")
print(df.nlargest(5, "Radiation")[["DateTime", "Radiation", "Temperature", "Humidity"]])
counts = df.groupby("Date")["Radiation"].count()
day_avg = df.groupby("Date")["Radiation"].mean()[counts >= 250]
cloudy_day = day_avg.idxmin()
clear_day = day_avg.idxmax()
plt.figure(figsize=(12, 5))
for day, colour in [(clear_day, "orange"), (cloudy_day, "grey")]:
    d = df[df["Date"] == day]
    plt.plot(d["DateTime"].dt.hour + d["DateTime"].dt.minute / 60, d["Radiation"],
             color=colour, label=str(day))
plt.title("Clearest day vs cloudiest day")
plt.xlabel("Hour of day")
plt.ylabel("Radiation (W/m²)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/B4_clear_vs_cloudy.png")
plt.show()
ts = df.set_index("DateTime")[["Radiation", "Temperature"]].resample("10min").mean().interpolate()
shifts = np.arange(0, 25)                      
corrs = [ts["Radiation"].corr(ts["Temperature"].shift(-k)) for k in shifts]
best_shift = shifts[np.argmax(corrs)] * 10    
plt.figure(figsize=(9, 4.5))
plt.plot(shifts * 10, corrs)
plt.axvline(best_shift, color="red", linestyle="--", label=f"Best match: {best_shift} min")
plt.title("How well temperature matches radiation after a delay")
plt.xlabel("Delay applied to temperature (minutes)")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.savefig("plots/B_bonus_temperature_lag.png")
plt.show()
print(f"ANSWER B BONUS: Temperature follows radiation with a delay of about {best_shift} minutes "
      f"(correlation rises from {corrs[0]:.3f} with no delay to {max(corrs):.3f}).")
print(f"  B1 Peak time            : {peak_hour}:00–{peak_hour + 1}:00 ({peak_value:.1f} W/m²)")
print(f"  B2 Strongest parameter  : {strongest} (r = {corr_all[strongest]:.2f})")
print(f"  B4 Midday dips (clouds) : {len(dips)} readings on {dips['Date'].nunique()} days")
print(f"  Bonus temperature lag   : about {best_shift} minutes")
print(df[["Data", "Time", "DateTime"]].head())
print("\nColumn types:")
print(df[["Data", "Time", "DateTime", "SunRise", "SunSet"]].dtypes)
daily = df.groupby("Date").agg(
    MaxRadiation=("Radiation", "max"),
    AvgRadiation=("Radiation", "mean"),
    Readings=("Radiation", "count"),    
    SunRise=("SunRise", "first"),
    SunSet=("SunSet", "first"),
)
daily["AvgRadiation"] = np.round(daily["AvgRadiation"], 2)
print(f"{len(daily)} days in the report")
print(daily.head(10))
daily.to_csv("daily_solar_report.csv")    
print("\nSaved the day-wise report to daily_solar_report.csv")
print(f"Days with fewer than 250 readings: {(daily['Readings'] < 250).sum()}")
plt.figure(figsize=(14, 5))
plt.plot(pd.to_datetime(daily.index), daily["MaxRadiation"], color="orangered", label="Daily maximum")
plt.plot(pd.to_datetime(daily.index), daily["AvgRadiation"], color="steelblue", label="Daily average")
plt.title("Daily maximum and average radiation")
plt.xlabel("Date")
plt.ylabel("Radiation (W/m^2)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/C2_daily_max_avg.png")
plt.show()
best_day = daily["MaxRadiation"].idxmax()
print(f"Date with the highest radiation: {best_day}  "
      f"({daily.loc[best_day, 'MaxRadiation']} W/m^2)")

day_df = df[df["Date"] == best_day].reset_index(drop=True)
peak_position = np.argmax(day_df["Radiation"].to_numpy())
peak = day_df.iloc[peak_position]
sunrise = peak["SunRise"]
sunset = peak["SunSet"]
after_sunrise = peak["DateTime"] - sunrise
before_sunset = sunset - peak["DateTime"]
solar_noon = sunrise + (sunset - sunrise) / 2
mins_after = int(after_sunrise.total_seconds() // 60)
mins_before = int(before_sunset.total_seconds() // 60)
diff_from_noon = (peak["DateTime"] - solar_noon).total_seconds() / 60
plt.figure(figsize=(12, 5))
plt.plot(day_df["DateTime"], day_df["Radiation"], color="orange")
plt.axvline(sunrise, color="gold", linestyle="--", label=f"Sunrise {sunrise:%H:%M}")
plt.axvline(sunset, color="purple", linestyle="--", label=f"Sunset {sunset:%H:%M}")
plt.scatter(peak["DateTime"], peak["Radiation"], color="red", s=80, zorder=5,
            label=f"Peak {peak['Radiation']:.0f} W/m² at {peak['DateTime']:%H:%M}")
plt.title(f"Radiation on {best_day} (the highest-radiation day)")
plt.xlabel("Time")
plt.ylabel("Radiation (W/m^2)")
plt.legend()
plt.tight_layout()
plt.show()
peak_rows = df.loc[df.groupby("Date")["Radiation"].idxmax()]
hours_after = (peak_rows["DateTime"] - peak_rows["SunRise"]).dt.total_seconds() / 3600
print(f"\nAcross all {len(peak_rows)} days the peak comes on average "
      f"{hours_after.mean():.2f} hours after sunrise.")
top5 = daily.nlargest(5, "AvgRadiation")[["AvgRadiation", "MaxRadiation", "Readings"]]
print("Top 5 days by average radiation:")
print(top5)
plt.figure(figsize=(9, 4.5))
plt.bar([str(d) for d in top5.index], top5["AvgRadiation"], color="orange")
plt.title("Top 5 days by average radiation")
plt.ylabel("Average radiation (W/m²)")
plt.tight_layout()
plt.savefig("plots/C6_top5_days.png")
plt.show()
print(f"  Highest-radiation day : {best_day}")
print(f"  Peak                  : {peak['Radiation']} W/m² at {peak['DateTime']:%H:%M:%S}")
print(f"  Conditions at peak    : {peak['Temperature']} °F, {peak['Humidity']} % humidity, "
      f"{peak['Speed']} mph wind")
print(f"  Peak after sunrise    : {mins_after // 60} h {mins_after % 60} min")
print(f"  Top 5 days by average : {', '.join(str(d) for d in top5.index)}")