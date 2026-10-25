# Solar Radiation Analysis and Prediction

STAC Wars – PS 2

This project looks at solar observation data logged by the Voyage-123 probe and tries to answer one question: what is the sun actually doing, and can we predict it? It covers data visualisation, inferences, a day-wise solar report and a machine learning model that predicts solar radiation.

## Dataset

`SolarPrediction.csv` has 32,686 readings taken roughly every 5 minutes from September to December 2016.

| Column | Meaning |
|---|---|
| UNIXTime | Timestamp in Unix format |
| Data | Date of the reading |
| Time | Time of the reading |
| Radiation | Solar radiation (W/m²) |
| Temperature | Temperature (°F) |
| Pressure | Air pressure (inHg) |
| Humidity | Humidity (%) |
| WindDirection(Degrees) | Wind direction |
| Speed | Wind speed (mph) |
| TimeSunRise | Sunrise time for that day |
| TimeSunSet | Sunset time for that day |

The dataset is not included in this repo. Download it from the Google Drive link given in the problem statement and place it in the same folder as the script.

## Project structure

```
├── solar_analysis.py          main script
├── SolarPrediction.csv        dataset (download separately)
├── daily_solar_report.csv     day-wise report created by the script
├── plots/                     all saved graphs
├── report.pdf                 final report
└── README.md
```

## What the script does

### Part A – Data Visualisation

- Radiation over the full period (hourly and daily average)
- Radiation vs Temperature, Humidity, Pressure and Wind Speed, with trend lines and correlation values
- Average radiation for each hour of the day
- Box plots to find outliers
- Bonus: heatmap of radiation by hour and date

### Part B – Inferences

- Time of day when radiation is highest
- Which parameter has the strongest link with radiation
- How radiation, temperature and humidity relate to each other
- Outliers and what could explain them
- Bonus: temperature lags behind radiation

### Part C – Solar Day Analysis

- Combined the Data and Time columns into one datetime column
- Daily maximum and average radiation
- Date with the highest radiation
- Time of the peak on that day, with temperature, humidity and wind speed
- How long after sunrise the peak happened
- Top 5 days by average radiation

### Bonus – Radiation Prediction

- Features: hour, minute, day of year, temperature, pressure, humidity, wind direction, wind speed, day length and minutes from solar noon
- 80/20 train-test split
- Models: Linear Regression and Random Forest
- Evaluated with RMSE and R²

## Key results

| Finding | Result |
|---|---|
| Peak hour | 12:00 – 13:00 (average 727.5 W/m²) |
| Strongest parameter | Temperature (r = 0.73) |
| Humidity vs radiation | r = -0.23 |
| Highest radiation day | 2016-09-04 |
| Peak on that day | 1601.26 W/m² at 12:15 |
| Conditions at peak | 61 °F, 93 % humidity, 9 mph wind |
| Peak after sunrise | 6 h 7 min |
| Top 5 days (avg) | 19 Sep, 2 Sep, 22 Sep, 25 Sep, 28 Sep |
| Temperature lag | about 30 minutes |

### Model scores

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 166.96 | 0.713 |
| Random Forest | 76.39 | 0.940 |

Random Forest did much better. The most useful feature was minutes from solar noon, followed by humidity. The sun's position decides how much radiation is possible, and humidity (a sign of clouds) decides how much of it actually reaches the ground.

## Some interesting things we found

- Radiation was never exactly zero at night. It stayed around 1 to 3 W/m², which is most likely sensor noise.
- The highest readings (above 1400 W/m²) all happened on humid days. This is probably cloud enhancement, where sunlight reflecting off the edges of clouds adds to the direct sunlight for a short time.
- Temperature does not rise at the same moment as radiation. It follows about 30 minutes later, because the air takes time to heat up.
- Radiation slowly drops from September to December as the days get shorter.
