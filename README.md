# InstaScribe

InstaScribe is a Streamlit dashboard for influencer intelligence, lead scoring, and admin-based account management. It combines profile-level metrics, post analytics, lead prioritisation, user authentication, and a searchable post inspector in a dark, card-based UI.

## Features

- Executive overview with KPI cards, smart insights, engagement trend, and lead-quality breakdowns
- Lead intelligence views for followers, engagement rate, authenticity, and ranked influencer profiles
- Post analytics with monthly engagement, likes vs comments charts, heatmaps, hashtag cloud, and post inspection
- AI lead scoring with pipeline funnel, score distribution, category comparison, and exportable outreach lists
- AI Insights with Groq-powered Q&A for whole-system summaries plus Post ID and Handle lookups
- Sidebar filters for category, quality, engagement rate, followers, follower tier, date range, and year
- Login, sign up, forgot password, and admin user-management flows

## Data Files

The app reads CSV files from the `data/` folder:

- `influencer_master.csv`
- `post_metrics.csv`
- `category_dim.csv`
- `lead_scoring.csv`
- `date_dim.csv` for reference or export, if needed

The authentication data is stored in `Supabase cloud database` and mirrored to `auth_store.yaml` for bootstrap and backup.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- plotly
- wordcloud
- PyYAML
- groq

## Run the App

```bash
streamlit run app.py
```

If you are using the virtual environment in this project on Windows:

```powershell
& .\myenv\Scripts\Activate.ps1
streamlit run app.py
```

## Safe Deployment and Ops Checklist

### Before Pushing to GitHub

- Do not commit real credentials. If `auth_store.yaml` contains production users, keep it out of the repo and provide `auth_store.example.yaml` instead.
- Keep `.streamlit/secrets.toml` out of the repo.
- Commit `app.py`, `requirements.txt`, `README.md`, and only the `data/` CSVs that are safe to share.

### Before Deploying

- Set the cookie secret in Streamlit secrets as `auth.cookie_key` (see code in `app.py`).
- Ensure the host provides persistent disk storage if you rely on `auth_store.db` (SQLite is not durable on many ephemeral hosts).
- Test the full auth loop after deploy: sign up, login, password reset, admin edit, and delete.

### Deployment Notes

- SQLite is fine for demos and single-instance internal apps.
- For production or multi-instance deployments, migrate auth to a managed DB (Postgres, MySQL, etc.).
- Use environment/secret management for all keys and credentials; never hardcode production secrets.

## Project Structure

```text
Project/
	app.py
	auth_supabase.py
	auth_store.yaml
	requirements.txt
	.streamlit/
		secrets.toml
	data/
		category_dim.csv
		date_dim.csv
		influencer_master.csv
		lead_scoring.csv
		post_metrics.csv
	.env
```

## What the Dashboard Calculates

- `FF Ratio` = `Following_Count / Follower_Count`
- `Lead Score` is computed from engagement rate, follower count, sentiment, and SaaS relevance, then normalised to a 0-100 scale
- Lead quality tiers are assigned from the lead score: `low`, `medium`, and `high`

## Notes

- The app automatically looks for the CSV files in `data/` relative to the script location.
- If the dashboard does not start, make sure the required CSV files are present and the dependencies are installed.
- If the auth page resets after deployment, check whether the host keeps local files between restarts; many platforms do not.
