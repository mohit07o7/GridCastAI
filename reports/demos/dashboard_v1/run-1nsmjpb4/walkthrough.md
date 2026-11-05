# GridCastAI local demonstration

Start with ./scripts/start_dashboard.sh from Desktop/GridCastAI.

1. Normal forecast: June 24, 2025 at 23:00 produces June 25’s 24 hours. Actuals are unavailable beyond the source history.
2. Historical fallback: January 9 at 15:00 shows explicitly labelled baseline fallbacks. Target quality remains unknown at issue time.
3. Synthetic blocked wind: the source frame is copied in memory, wind is made invalid, and demand/solar continue. Missing wind and net-demand chart points are gaps, never zeros. No source file changes.
4. Historical accuracy: choose net demand, horizon 24, delivered cohort, all months. Model and baseline use identical paired rows.
5. Quality audit: choose April 2025 in Historical accuracy and open Quality and exclusions. Input problems and retrospective target exclusions are separate.
6. Optimization status: Deferred. No battery schedule or savings is generated.

Downloads contain the same forecast rows and original JSON as the app. Results are retrospective demonstrations, not live forecasts actually issued in 2025.
