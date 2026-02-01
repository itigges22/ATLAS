"""
ATLAS Dashboard - Monitoring and control interface.
"""

import os
import redis
import json
from datetime import date, datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="ATLAS Dashboard")

templates = Jinja2Templates(directory="templates")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    # Get queue stats
    queue_stats = {
        "p0": redis_client.llen("tasks:p0"),
        "p1": redis_client.llen("tasks:p1"),
        "p2": redis_client.llen("tasks:p2"),
    }

    # Get today's metrics (use UTC to match task-worker)
    today = datetime.now(timezone.utc).date().isoformat()
    daily_stats = redis_client.hgetall(f"metrics:daily:{today}")

    # Get recent tasks
    recent_tasks = redis_client.lrange("metrics:recent_tasks", 0, 19)
    recent_tasks = [json.loads(t) for t in recent_tasks]

    # Get weekly trend (use UTC)
    weekly_trend = []
    utc_today = datetime.now(timezone.utc).date()
    for i in range(7):
        day = (utc_today - timedelta(days=i)).isoformat()
        stats = redis_client.hgetall(f"metrics:daily:{day}")
        weekly_trend.append({
            "date": day,
            "total": int(stats.get("tasks_total", 0)),
            "success": int(stats.get("tasks_success", 0))
        })
    weekly_trend.reverse()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "queue_stats": queue_stats,
        "daily_stats": daily_stats,
        "recent_tasks": recent_tasks,
        "weekly_trend": weekly_trend
    })

@app.get("/api/stats")
async def api_stats():
    """API endpoint for stats (for AJAX refresh)."""
    today = datetime.now(timezone.utc).date().isoformat()

    return {
        "queue": {
            "p0": redis_client.llen("tasks:p0"),
            "p1": redis_client.llen("tasks:p1"),
            "p2": redis_client.llen("tasks:p2"),
        },
        "daily": redis_client.hgetall(f"metrics:daily:{today}"),
        "recent": [
            json.loads(t)
            for t in redis_client.lrange("metrics:recent_tasks", 0, 9)
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
