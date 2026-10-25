import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("SolarPrediction.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

date_part = df["Data"].str.split().str[0]
fmt = "%m/%d/%Y %H:%M:%S"
df["DateTime"] = pd.to_datetime(date_part + " " + df["Time"], format=fmt)
df["SunRise"] = pd.to_datetime(date_part + " " + df["TimeSunRise"], format=fmt)
df["SunSet"] = pd.to_datetime(date_part + " " + df["TimeSunSet"], format=fmt)

df = df.sort_values("DateTime").reset_index(drop=True)
df = df.rename(columns={"WindDirection(Degrees)": "WindDirection"})
df["Date"] = df["DateTime"].dt.date
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month_name()
df["IsDaylight"] = df["DateTime"].between(df["SunRise"], df["SunSet"])

print("Data covers", df["DateTime"].min(), "to", df["DateTime"].max())

daily_mean = df.groupby("Date")["Radiation"].mean()
first_days = df[df["DateTime"] < "2016-09-06"]
hourly = df.groupby("Hour")["Radiation"].mean()
sample = df.sample(5000, random_state=42)

num_cols = ["Radiation", "Temperature", "Pressure", "Humidity", "WindDirection", "Speed"]
corr = df[num_cols].corr()
print("Correlation with Radiation:")
print(corr["Radiation"].drop("Radiation").sort_values(ascending=False).round(3))

box_cols = ["Radiation", "Temperature", "Humidity", "Pressure", "Speed"]
scaled = (df[box_cols] - df[box_cols].mean()) / df[box_cols].std()
daytime = df[(df["Hour"] >= 6) & (df["Hour"] < 18)]

peak_hour = hourly.idxmax()
peak_value = hourly.max()
print("\nAverage radiation per hour:")
print(hourly.round(1))
print("Top 3 hours:")
print(hourly.sort_values(ascending=False).head(3).round(1))
print(f"B1: Radiation is highest between {peak_hour}:00 and {peak_hour + 1}:00 "
      f"(avg {peak_value:.1f} W/m^2)")

params = ["Temperature", "Pressure", "Humidity", "WindDirection", "Speed"]
corr_all = df[params + ["Radiation"]].corr()["Radiation"].drop("Radiation")
day_rows = df[df["IsDaylight"]]
corr_day = day_rows[params + ["Radiation"]].corr()["Radiation"].drop("Radiation")
corr_table = pd.DataFrame({"All readings": corr_all, "Daytime only": corr_day})
corr_table = corr_table.loc[corr_all.abs().sort_values(ascending=False).index]
print(corr_table.round(3))

strongest = corr_table.index[0]
print(f"B2: {strongest} has the strongest link with radiation (r = {corr_all[strongest]:.2f}). "
      f"Sunlight heats the air, so more radiation means a higher temperature.")

trio = df[["Radiation", "Temperature", "Humidity"]].corr().round(2)
print(trio)

df["TempBand"] = pd.cut(df["Temperature"], bins=[30, 40, 45, 50, 55, 60, 75])
bands = df.groupby("TempBand", observed=True)[["Radiation", "Humidity"]].mean().round(1)
print("\nAverage radiation and humidity per temperature band:")
print(bands)

print(f"B3: Radiation rises with temperature (r = {trio.loc['Radiation', 'Temperature']}) "
      f"and falls with humidity (r = {trio.loc['Radiation', 'Humidity']}). "
      f"Temperature and humidity move against each other (r = {trio.loc['Temperature', 'Humidity']}), "
      f"so hot, dry hours get the most sun and cool, humid hours the least.")

print("\nB4: Outliers per column (1.5 x IQR rule)")
for col in box_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"   {col:12s}: {n}")

hour_median = df.groupby("Hour")["Radiation"].transform("median")
dips = df[df["Hour"].between(10, 14) & (df["Radiation"] < 0.3 * hour_median)]
print(f"Midday readings under 30% of normal: {len(dips)} readings on {dips['Date'].nunique()} days")

night = df[~df["IsDaylight"] & (df["Radiation"] > 10)]
print(f"Readings above 10 W/m^2 outside daylight: {len(night)}")

print("Five highest radiation readings:")
print(df.nlargest(5, "Radiation")[["DateTime", "Radiation", "Temperature", "Humidity"]])

print("B4 explanation:")
print("   - Midday dips are most likely clouds passing over the sensor.")
print("   - Night readings near 1-3 W/m^2 are sensor noise; anything higher sits right at")
print("     sunrise/sunset, so it is twilight light rather than an error.")
print("   - Speed and pressure outliers point to gusty or stormy spells.")
print("   - The highest readings (1400-1600 W/m^2) all come on humid days (86-98 %).")
print("     This looks like cloud enhancement: sunlight bouncing off cloud edges adds")
print("     to the direct beam for a few minutes, pushing radiation above normal.")

counts = df.groupby("Date")["Radiation"].count()
full_days = df.groupby("Date")["Radiation"].mean()[counts >= 250]
cloudy_day = full_days.idxmin()
clear_day = full_days.idxmax()

ts = (df.set_index("DateTime")[["Radiation", "Temperature"]]
        .resample("10min").mean().interpolate())
lags = np.arange(25)
lag_corr = [ts["Radiation"].corr(ts["Temperature"].shift(-k)) for k in lags]
best_lag = lags[np.argmax(lag_corr)] * 10
print(f"B bonus: Temperature trails radiation by about {best_lag} minutes "
      f"(r goes from {lag_corr[0]:.3f} with no delay to {max(lag_corr):.3f}).")

print(df[["Data", "Time", "DateTime"]].head())
print(df[["DateTime", "SunRise", "SunSet"]].dtypes)

daily = df.groupby("Date").agg(
    MaxRadiation=("Radiation", "max"),
    AvgRadiation=("Radiation", "mean"),
    Readings=("Radiation", "count"),
    SunRise=("SunRise", "first"),
    SunSet=("SunSet", "first"),
)
daily["AvgRadiation"] = daily["AvgRadiation"].round(2)
print(f"{len(daily)} days in the report")
print(daily.head(10))
daily.to_csv("daily_solar_report.csv")
print("Saved daily_solar_report.csv")
print(f"Days with under 250 readings: {(daily['Readings'] < 250).sum()}")

best_day = daily["MaxRadiation"].idxmax()
print(f"Highest radiation date: {best_day} ({daily.loc[best_day, 'MaxRadiation']} W/m^2)")

day_df = df[df["Date"] == best_day].reset_index(drop=True)
peak = day_df.iloc[np.argmax(day_df["Radiation"].to_numpy())]
sunrise, sunset = peak["SunRise"], peak["SunSet"]
solar_noon = sunrise + (sunset - sunrise) / 2

mins_after = int((peak["DateTime"] - sunrise).total_seconds() // 60)
mins_before = int((sunset - peak["DateTime"]).total_seconds() // 60)
noon_gap = (peak["DateTime"] - solar_noon).total_seconds() / 60

peak_rows = df.loc[df.groupby("Date")["Radiation"].idxmax()]
hrs_after = (peak_rows["DateTime"] - peak_rows["SunRise"]).dt.total_seconds() / 3600
print(f"Across all {len(peak_rows)} days the peak comes {hrs_after.mean():.2f} hours after sunrise on average.")

top5 = daily.nlargest(5, "AvgRadiation")[["AvgRadiation", "MaxRadiation", "Readings"]]
print("Top 5 days by average radiation:")
print(top5)



def all_graphs():
    while True:
        print("\nWhich graph do you want to see?")
        print(" 1 = Radiation timeline")
        print(" 2 = First 5 days")
        print(" 3 = Hourly average")
        print(" 4 = Hourly by month")
        print(" 5 = Scatter plots")
        print(" 6 = Correlation heatmap")
        print(" 7 = Boxplot by hour")
        print(" 8 = Boxplot all parameters")
        print(" 9 = Histograms")
        print("10 = Peak hour")
        print("11 = Correlation bars")
        print("12 = Temperature vs humidity")
        print("13 = Clear vs cloudy day")
        print("14 = Temperature lag")
        print("15 = Daily max and average")
        print("16 = Best day")
        print("17 = Top 5 days")
        print("18 = Show all")
        print(" 0 = Next part")
        print("19 = Exit")
        ask = input("Enter your choice : ").strip()

        if ask == "0":
            return True
        elif ask == "19":
            print("Bye!")
            return False
        elif ask == "18":
            for k in range(1, 18):
                show_graph(str(k))
        elif not show_graph(ask):
            print("Invalid choice.")


def show_graph(ask):
    if ask == "1":
        plt.figure(figsize=(14, 5))
        plt.plot(df["DateTime"], df["Radiation"], lw=0.3, alpha=0.6, label="All readings")
        plt.plot(pd.to_datetime(daily_mean.index), daily_mean.values, color="red", lw=2, label="Daily average")
        plt.title("Solar radiation over the full period")
        plt.xlabel("Date")
        plt.ylabel("Radiation (W/m^2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/A1_radiation_timeline.png")
        plt.show()
    elif ask == "2":
        plt.figure(figsize=(14, 4))
        plt.plot(first_days["DateTime"], first_days["Radiation"], color="orange")
        plt.title("Radiation over the first 5 days")
        plt.xlabel("Date and time")
        plt.ylabel("Radiation (W/m^2)")
        plt.tight_layout()
        plt.savefig("plots/A1_first_5_days.png")
        plt.show()
    elif ask == "3":
        plt.figure(figsize=(10, 5))
        plt.plot(hourly.index, hourly.values, marker="o", color="darkorange")
        plt.title("Average radiation by hour of day")
        plt.xlabel("Hour")
        plt.ylabel("Average radiation (W/m^2)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig("plots/A1_hourly_average.png")
        plt.show()
    elif ask == "4":
        plt.figure(figsize=(10, 5))
        by_month = df.groupby(["Month", "Hour"])["Radiation"].mean()
        for m in df["Month"].unique():
            plt.plot(by_month[m].index, by_month[m].values, label=m)
        plt.legend(title="Month")
        plt.title("Average radiation by hour, per month")
        plt.xlabel("Hour")
        plt.ylabel("Average radiation (W/m^2)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig("plots/A1_hourly_by_month.png")
        plt.show()
    elif ask == "5":
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, col in zip(axes.flat, ["Temperature", "Humidity", "Pressure", "Speed"]):
            ax.scatter(sample[col], sample["Radiation"], s=5, alpha=0.3)
            slope, intercept = np.polyfit(sample[col], sample["Radiation"], 1)
            xs = np.linspace(sample[col].min(), sample[col].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="red")
            ax.set_xlabel(col)
            ax.set_ylabel("Radiation")
            r = df[col].corr(df["Radiation"])
            ax.set_title(f"Radiation vs {col} (r = {r:.2f})")
        plt.tight_layout()
        plt.savefig("plots/A2_scatter_plots.png")
        plt.show()
    elif ask == "6":
        plt.figure(figsize=(8, 6))
        my_cmap = LinearSegmentedColormap.from_list("mine", ["#3b6e8f", "#9cc3d5", "#f4efe6", "#e8b27d", "#b5523b"])
        n = len(corr)
        img = plt.imshow(corr, cmap=my_cmap, vmin=-1, vmax=1)
        plt.colorbar(img, shrink=0.8, label="r")
        plt.xticks(range(n), corr.columns, rotation=90)
        plt.yticks(range(n), corr.index)
        for a in range(n):
            for b in range(n):
                v = corr.iloc[a, b]
                plt.text(b, a, f"{v:.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.6 else "black")
        ax = plt.gca()
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        plt.title("Correlation between parameters")
        plt.tight_layout()
        plt.savefig("plots/A2_correlation_heatmap.png")
        plt.show()
    elif ask == "7":
        plt.figure(figsize=(14, 5))
        hours = sorted(df["Hour"].unique())
        plt.boxplot([df.loc[df["Hour"] == h, "Radiation"] for h in hours], positions=range(len(hours)),
                    patch_artist=True, boxprops={"facecolor": "orange"}, medianprops={"color": "black"},
                    flierprops={"markersize": 3})
        plt.xticks(range(len(hours)), hours)
        plt.title("Radiation spread at each hour")
        plt.xlabel("Hour")
        plt.ylabel("Radiation (W/m^2)")
        plt.tight_layout()
        plt.savefig("plots/A3_boxplot_by_hour.png")
        plt.show()
    elif ask == "8":
        plt.figure(figsize=(10, 5))
        boxes = plt.boxplot([scaled[c] for c in box_cols], patch_artist=True,
                            medianprops={"color": "black"}, flierprops={"markersize": 2})
        for box, colour in zip(boxes["boxes"], plt.rcParams["axes.prop_cycle"].by_key()["color"]):
            box.set_facecolor(colour)
        plt.xticks(range(1, len(box_cols) + 1), box_cols)
        plt.title("All parameters on one scale (dots are outliers)")
        plt.ylabel("Std deviations from mean")
        plt.tight_layout()
        plt.savefig("plots/A3_boxplot_all_parameters.png")
        plt.show()
    elif ask == "9":
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].hist(df["Radiation"], bins=60, edgecolor="white")
        axes[0].set_ylabel("Count")
        axes[0].set_title("All readings")
        axes[0].set_xlabel("Radiation (W/m^2)")
        axes[1].hist(daytime["Radiation"], bins=60, color="orange", edgecolor="white")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Daytime only (6am to 6pm)")
        axes[1].set_xlabel("Radiation (W/m^2)")
        plt.tight_layout()
        plt.savefig("plots/A3_histograms.png")
        plt.show()
    elif ask == "10":
        bar_colours = ["#23ebe1" if h == peak_hour else "#eb3e0e" for h in hourly.index]
        plt.figure(figsize=(11, 5))
        plt.bar(hourly.index, hourly.values, color=bar_colours)
        plt.title("Average radiation by hour (highlighted = peak)")
        plt.xlabel("Hour")
        plt.ylabel("Average radiation (W/m^2)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig("plots/B1_peak_hour.png")
        plt.show()
    elif ask == "11":
        corr_table.plot(kind="barh", figsize=(11, 5), color=["#1d4ade", "#f20511"])
        plt.axvline(0, color="black", lw=1)
        plt.title("Correlation of each parameter with radiation")
        plt.xlabel("r")
        plt.tight_layout()
        plt.savefig("plots/B2_correlations.png")
        plt.show()
    elif ask == "12":
        plt.figure(figsize=(11, 6))
        pts = plt.scatter(sample["Temperature"], sample["Humidity"], c=sample["Radiation"],
                          cmap=LinearSegmentedColormap.from_list("sun", ["#5b7a99", "#a3b18a", "#e9c46a", "#e76f51"]),
                          s=10, alpha=0.6, edgecolors="none")
        plt.colorbar(pts, label="Radiation (W/m^2)")
        plt.title("Temperature vs humidity, coloured by radiation")
        plt.xlabel("Temperature (°F)")
        plt.ylabel("Humidity (%)")
        plt.tight_layout()
        plt.savefig("plots/B3_temp_humidity_radiation.png")
        plt.show()
    elif ask == "13":
        plt.figure(figsize=(12, 5))
        for day, colour in [(clear_day, "orange"), (cloudy_day, "grey")]:
            d = df[df["Date"] == day]
            plt.plot(d["DateTime"].dt.hour + d["DateTime"].dt.minute / 60, d["Radiation"],
                     color=colour, label=str(day))
        plt.title("Clearest day vs cloudiest day")
        plt.xlabel("Hour")
        plt.ylabel("Radiation (W/m^2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/B4_clear_vs_cloudy.png")
        plt.show()
    elif ask == "14":
        plt.figure(figsize=(9, 4.5))
        plt.plot(lags * 10, lag_corr)
        plt.axvline(best_lag, color="red", ls="--", label=f"Best match: {best_lag} min")
        plt.title("Temperature vs radiation at different delays")
        plt.xlabel("Delay on temperature (min)")
        plt.ylabel("Correlation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/B_bonus_temperature_lag.png")
        plt.show()
    elif ask == "15":
        plt.figure(figsize=(14, 5))
        plt.plot(pd.to_datetime(daily.index), daily["MaxRadiation"], color="orangered", label="Daily max")
        plt.plot(pd.to_datetime(daily.index), daily["AvgRadiation"], color="steelblue", label="Daily average")
        plt.title("Daily maximum and average radiation")
        plt.xlabel("Date")
        plt.ylabel("Radiation (W/m^2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/C2_daily_max_avg.png")
        plt.show()
    elif ask == "16":
        plt.figure(figsize=(12, 5))
        plt.plot(day_df["DateTime"], day_df["Radiation"], color="orange")
        plt.axvline(sunrise, color="gold", ls="--", label=f"Sunrise {sunrise:%H:%M}")
        plt.axvline(sunset, color="purple", ls="--", label=f"Sunset {sunset:%H:%M}")
        plt.scatter(peak["DateTime"], peak["Radiation"], color="red", s=80, zorder=5,
                    label=f"Peak {peak['Radiation']:.0f} W/m^2 at {peak['DateTime']:%H:%M}")
        plt.title(f"Radiation on {best_day}")
        plt.xlabel("Time")
        plt.ylabel("Radiation (W/m^2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/C5_best_day.png")
        plt.show()
    elif ask == "17":
        plt.figure(figsize=(9, 4.5))
        plt.bar([str(d) for d in top5.index], top5["AvgRadiation"], color="orange")
        plt.title("Top 5 days by average radiation")
        plt.ylabel("Average radiation (W/m^2)")
        plt.tight_layout()
        plt.savefig("plots/C6_top5_days.png")
        plt.show()
    else:
        return False
    return True


units = {"Radiation": "W/m^2", "Temperature": "°F", "Pressure": "inHg", "Humidity": "%",
         "WindDirection": "°", "Speed": "mph"}
ts_df = df.set_index("DateTime")


def label(col):
    return f"{col} ({units[col]})" if col in units else col


def graph(i1, i2="Radiation"):
    hourly_1 = ts_df[i1].resample("h").mean().interpolate()
    hourly_2 = ts_df[i2].resample("h").mean().interpolate()
    daily_1 = ts_df[i1].resample("D").mean().interpolate()
    daily_2 = ts_df[i2].resample("D").mean().interpolate()

    while True:
        print("\nWhich graph do you want?")
        print("1 = Line graph, 2 = Scatter, 3 = Custom axis, 4 = Radiation daily and hourly curve, 5 = Exit")
        ask = input("Enter 1/2/3/4/5 : ").strip()

        if ask == "1":
            fig, ax1 = plt.subplots(figsize=(16, 6))
            ax2 = ax1.twinx()
            ax1.plot(hourly_1.index, hourly_1.values, color="#c0504d", lw=0.4, alpha=0.25)
            ax1.plot(daily_1.index, daily_1.values, color="#c0504d", lw=2, label=f"{i1} (daily avg)")
            ax2.plot(hourly_2.index, hourly_2.values, color="#e9a23b", lw=0.4, alpha=0.25)
            ax2.plot(daily_2.index, daily_2.values, color="#3b6e8f", lw=2, label=f"{i2} (daily avg)")
            ax1.set_xlabel("Date")
            ax1.set_ylabel(label(i1), color="#c0504d")
            ax2.set_ylabel(label(i2), color="#3b6e8f")
            ax2.grid(False)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right")
            plt.title(f"{i1} vs {i2} over time")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()
        elif ask == "2":
            plt.figure(figsize=(10, 6))
            sc = plt.scatter(df[i1], df[i2], c=df["Hour"], cmap="twilight", s=4, alpha=0.5)
            plt.colorbar(sc, label="Hour of day")
            r = df[i1].corr(df[i2])
            plt.title(f"{i2} vs {i1} (r = {r:.2f})")
            plt.xlabel(label(i1))
            plt.ylabel(label(i2))
            plt.tight_layout()
            plt.show()
        elif ask == "3":
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "UNIXTime"]
            print(f"Columns : {cols}")
            x = input("Enter the x-axis column: ").strip()
            y = input("Enter the y-axis column: ").strip()
            if x in cols and y in cols:
                plt.figure(figsize=(10, 6))
                plt.scatter(df[x], df[y], s=4, alpha=0.4, color="#5b7a99")
                plt.xlabel(label(x))
                plt.ylabel(label(y))
                plt.title(f"{y} vs {x} (r = {df[x].corr(df[y]):.2f})")
                plt.tight_layout()
                plt.show()
            else:
                print("Invalid column name.")
        elif ask == "4":
            hourly0 = ts_df["Radiation"].resample("h").mean()
            daily0 = ts_df["Radiation"].resample("D").mean().interpolate()
            plt.figure(figsize=(14, 5))
            plt.plot(hourly0.index, hourly0.values, color="#c0504d", alpha=0.3, label="Hourly")
            plt.plot(daily0.index, daily0.values, color="#3b6e8f", alpha=0.8, label="Daily")
            plt.xlabel("Date")
            plt.ylabel(label("Radiation"))
            plt.title("Radiation, hourly and daily average")
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif ask == "5":
            break
        else:
            print("Invalid choice.")


if all_graphs():
    graph("Humidity")
