# MISSION BRIEF: Sprint 1 (Data Integration & Backend Bring-up)

## 🎯 Objective
Create APIs that return intersting stats from our 10-year eBird database.

## 🤖 Agent Task List

### Phase 1: Backend API Development (Backend Agent)
- [x] **Task 1: Setup.** Initialize a FastAPI application in `/backend`. Create a `requirements.txt` that includes `fastapi`, `uvicorn`, `supabase`.
- [x] **Task 2: The Endpoint.** Create a `GET /api/predictions` endpoint. It must accept: `latitude` (float), `longitude` (float), `month` (int, starts at 1), `radius_miles` (float, default 10), and `years_limit` (int).
- [x] **Task 3: Database Call.** Inside the endpoint, call the Supabase RPC `get_seasonal_hotspots(user_lat, user_lon, month, radius, years)`.
- [x] **Task 4: Response Formatting.** Ensure the JSON response perfectly matches the TypeScript interface expected by `results-list.tsx`.

### Phase 2: Backend API Development For Annual Summaries (Backend Agent)
- [x] **Task 1: The 2nd Endpoint.** Create a `GET /api/stats/annualsummary` endpoint. It does not require any parameter.
- [x] **Task 2: Database Call.** Inside the endpoint above, call the Supabase RPC `get_annual_sightings_summary()`.
- [x] **Task 3: The 3rd Endpoint.** Create a `GET /api/stats/state` endpoint. It must accept: `year` (int) as a parameter.
- [x] **Task 4: Database Call.** Inside the endpoint above, call the Supabase RPC `get_state_stats_by_year()`.
- [x] **Task 5: The 4th Endpoint.** Create a `GET /api/stats/counties` endpoint. It must accept: `state` (text) and `year` (int) as parameters.
- [x] **Task 6: Database Call.** Inside the endpoint above, call the Supabase RPC `get_county_stats_by_state_year()`.

### Phase 3: Backend API Development For Top Parks (Backend Agent)
- [x] **Task 1: The 5th Endpoint.** Create a `GET /api/stats/topparks` endpoint. It does not require any parameter.
- [x] **Task 2: Database Call.** Inside the endpoint above, call the Supabase RPC `get_top_20_parks()`.
- [x] **Task 3: Caching.** Since this data doesn't really change, cache the results for 24 hours.