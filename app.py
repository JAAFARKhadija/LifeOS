
import streamlit as st
from datetime import date
import pandas as pd
import plotly.express as px
from datetime import timedelta
from modules.activities import add_activity, get_all_activities
from modules.tasks import (
    add_task,
    get_all_tasks,
    update_task_status,
    delete_task
)
from modules.habits import (
    add_habit,
    get_all_habits,
    log_habit,
    get_habit_logs
)
from machine_learning.predict import predict_task_completion
from modules.ai_agent import chat_with_lifeos

# Page configuration
st.set_page_config(
    page_title="LifeOS",
    page_icon="🌱",
    layout="wide"
)

# Header
st.title("🌱 LifeOS")
st.subheader("Your Personal AI Assistant")


# ==========================================
# LIFEOS DASHBOARD
# ==========================================

st.divider()
st.header("📊 Your Personal Dashboard")

# Retrieve database information
dashboard_tasks = get_all_tasks()
dashboard_activities = get_all_activities()

# ==========================================
# AUTOMATED TASK REMINDERS
# ==========================================

st.subheader("🔔 Automatic Task Reminders")

today = date.today()

overdue_tasks = []
due_today_tasks = []
upcoming_tasks = []

for task in dashboard_tasks:

    task_id = task[0]
    title = task[1]
    due_date = task[5]
    status = task[7]

    if status != "Pending" or not due_date:
        continue

    deadline = date.fromisoformat(str(due_date)[:10])

    if deadline < today:
        overdue_tasks.append((title, deadline))

    elif deadline == today:
        due_today_tasks.append((title, deadline))

    elif (deadline - today).days <= 3:
        upcoming_tasks.append((title, deadline))


if overdue_tasks:
    st.error(f"🚨 {len(overdue_tasks)} overdue task(s)")

    for title, deadline in overdue_tasks:
        st.write(f"- **{title}** — Overdue since {deadline}")

if due_today_tasks:
    st.warning(f"⏰ {len(due_today_tasks)} task(s) due today")

    for title, deadline in due_today_tasks:
        st.write(f"- **{title}** — Due today")

if upcoming_tasks:
    st.info(f"📅 {len(upcoming_tasks)} upcoming deadline(s)")

    for title, deadline in upcoming_tasks:
        st.write(f"- **{title}** — Due {deadline}")

if not overdue_tasks and not due_today_tasks and not upcoming_tasks:
    st.success("You're all clear! No urgent deadlines.")

# Task statistics
total_tasks = len(dashboard_tasks)

completed_tasks = sum(
    1 for task in dashboard_tasks
    if task[7] == "Completed"
)

pending_tasks = sum(
    1 for task in dashboard_tasks
    if task[7] == "Pending"
)

# Activity statistics (duration is in minutes)
study_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Study"
)

gym_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Gym"
)

volleyball_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Volleyball"
)

# Convert minutes to hours
study_hours = round(study_minutes / 60, 1)
gym_hours = round(gym_minutes / 60, 1)
volleyball_hours = round(volleyball_minutes / 60, 1)

# Display task metrics
st.subheader("📚 Study Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Tasks", total_tasks)
col2.metric("Completed", completed_tasks)
col3.metric("Pending", pending_tasks)

# Display activity metrics
st.subheader("🏃 Activity Overview")

col1, col2, col3 = st.columns(3)

col1.metric("📖 Study Hours", f"{study_hours} h")
col2.metric("🏋️ Gym Hours", f"{gym_hours} h")
col3.metric("🏐 Volleyball Hours", f"{volleyball_hours} h")


# ==========================================
# WEEKLY ACTIVITY CHART
# ==========================================

st.subheader("📈 Weekly Activity Overview")

today = date.today()
start_date = today - timedelta(days=6)

weekly_records = []

for activity in dashboard_activities:

    activity_date = date.fromisoformat(activity[4])

    if start_date <= activity_date <= today:

        weekly_records.append({
            "Date": activity_date,
            "Activity": activity[1],
            "Hours": activity[3] / 60
        })

if weekly_records:

    weekly_df = pd.DataFrame(weekly_records)

    chart_data = (
        weekly_df
        .groupby(["Date", "Activity"], as_index=False)["Hours"]
        .sum()
    )

    fig = px.bar(
        chart_data,
        x="Date",
        y="Hours",
        color="Activity",
        barmode="group",
        title="Hours spent per activity (Last 7 Days)",
        labels={
            "Hours": "Duration (hours)",
            "Date": "Day"
        }
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Hours",
        legend_title="Activity",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No activities recorded during the last 7 days.")

st.divider()


st.divider()

st.header("📚 Study Task Manager")

# Add task form
with st.expander("➕ Add a New Task", expanded=True):

    with st.form("add_task_form"):

        title = st.text_input("Task Title")
        subject = st.text_input("Subject")
        task_type = st.selectbox("Task Type",
        ["Cours", "TD", "TP", "Projects", "Series", "Tests"])
        
        col1, col2, col3 = st.columns(3)

        with col1:
            due_date = st.date_input("Deadline", value=date.today())

        with col2:
            priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High"]
            )

        with col3:
            duration = st.number_input(
                "Estimated Duration (minutes)",
                min_value=0,
                value=60
            )

        submitted = st.form_submit_button("Add Task")

        if submitted:
            if title.strip():
                add_task(
                    title=title.strip(),
                    subject=subject.strip() or None,
                    task_type=task_type,
                    due_date=str(due_date),
                    priority=priority,
                    estimated_duration=duration
                )

                st.success("Task added successfully!")
                st.rerun()

            else:
                st.error("Please enter a task title.")


# Display tasks
st.divider()
st.subheader("📋 My Tasks")

tasks = get_all_tasks()

if not tasks:
    st.info("No tasks yet. Add your first task above!")

else:
    for task in tasks:

        task_id = task[0]
        title = task[1]
        subject = task[2]
        task_type = task[3]
        created_date = task[4]
        due_date = task[5]
        priority = task[6]
        status = task[7]
        duration = task[8]

        with st.container(border=True):

            col1, col2, col3 = st.columns([4, 2, 2])

            with col1:
                st.markdown(f"**{title}**")
                st.caption(f"Subject: {subject or 'N/A'}")
                st.caption(f"Type: {task_type}")
                st.caption(f"Created: {created_date}")
                st.caption(f"Deadline: {due_date}")

            with col2:
                st.write(f"Priority: {priority}")
                st.write(f"Duration: {duration or 0} min")

            with col3:
                st.write(f"Status: {status}")

                if status == "Pending":
                    if st.button(
                        "✅ Complete",
                        key=f"complete_{task_id}"
                    ):
                        update_task_status(task_id, "Completed")
                        st.rerun()

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{task_id}"
                ):
                    delete_task(task_id)
                    st.rerun()

            # ==========================================
            # ML TASK COMPLETION PREDICTION
            # ==========================================

            if status == "Pending" and created_date and due_date and duration is not None:

                try:
                    result = predict_task_completion(
                        subject=subject,
                        task_type=task_type,
                        priority=priority,
                        estimated_duration=duration,
                        created_date=created_date,
                        due_date=due_date
                    )

                    probability = result["on_time_probability"]

                    st.write("**ML Prediction (Experimental)**")

                    if result["prediction"] == 1:
                        st.success("Model estimates likely on-time completion")
                    else:
                        st.warning("Model flags potential delay")

                    st.progress(probability)

                    st.caption(
                        f"Estimated on-time score: {probability:.1%}"
                    )

                except Exception as e:
                    st.error(f"Prediction unavailable: {e}")



# ==========================================
# ACTIVITIES MANAGER
# ==========================================

st.divider()
st.header("🏃 Activities Manager")

with st.expander("➕ Record a New Activity", expanded=False):

    with st.form("activity_form"):

        activity_type = st.selectbox(
            "Activity Type",
            ["Study", "Gym", "Volleyball"]
        )

        title = st.text_input("Activity Title")

        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                value=60
            )

        with col2:
            activity_date = st.date_input(
                "Activity Date",
                value=date.today(),
                key="activity_date"
            )

        notes = st.text_area("Notes (optional)")

        submitted = st.form_submit_button("Save Activity")

        if submitted:
            if title.strip():
                add_activity(
                    activity_type,
                    title.strip(),
                    duration,
                    str(activity_date),
                    notes.strip() or None
                )

                st.success("Activity saved!")
                st.rerun()
            else:
                st.error("Please enter an activity title.")

# Display activities
st.subheader("📊 Activity History")

activities = get_all_activities()

if activities:

    for activity in activities:

        activity_id, activity_type, title, duration, activity_date, notes = activity

        with st.container(border=True):
            st.markdown(f"**{title}** — {activity_type}")
            st.write(f"📅 {activity_date} | ⏱️ {duration} minutes")

            if notes:
                st.caption(notes)

else:
    st.info("No activities recorded yet.")


# ==========================================
# HABIT TRACKER
# ==========================================

st.divider()
st.header("🎯 Daily Habit Tracker")

today = str(date.today())

# Add a new habit
with st.expander("➕ Create a New Habit"):

    with st.form("habit_form"):

        habit_name = st.text_input("Habit Name")
        description = st.text_input("Description")

        submitted = st.form_submit_button("Create Habit")

        if submitted:
            if habit_name.strip():
                try:
                    add_habit(
                        habit_name.strip(),
                        description.strip() or None
                    )
                    st.success("Habit created!")
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not create habit: {error}")
            else:
                st.warning("Enter a habit name.")

# Display habits
habits = get_all_habits()

st.subheader("My Daily Habits")

if not habits:
    st.info("No habits created yet.")

else:
    for habit in habits:

        habit_id = habit[0]
        name = habit[1]
        description = habit[2]

        logs = get_habit_logs(habit_id)

        completed_today = any(
            log[2] == today and log[3] == 1
            for log in logs
        )

        with st.container(border=True):

            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown(f"**{name}**")
                if description:
                    st.caption(description)

            with col2:

                if completed_today:
                    st.success("Completed today! ✅")

                    if st.button(
                        "Undo completion",
                        key=f"undo_{habit_id}"
                    ):
                        log_habit(habit_id, today, False)
                        st.rerun()

                else:
                    if st.button(
                        "Mark as completed",
                        key=f"complete_habit_{habit_id}"
                    ):
                        log_habit(habit_id, today, True)
                        st.rerun()

# ==========================================
# LIFEOS AI ASSISTANT
# ==========================================

st.divider()
st.header("🤖 LifeOS AI Assistant")

st.caption(
    "Ask questions about your study tasks, activities, "
    "habits and productivity."
)

# Initialize chat history
if "lifeos_chat_history" not in st.session_state:
    st.session_state.lifeos_chat_history = []

# Display previous messages
for item in st.session_state.lifeos_chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

# Chat input
if prompt := st.chat_input(
    "Ask LifeOS something..."
):

    # Save and display user message
    st.session_state.lifeos_chat_history.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("LifeOS is thinking..."):

            try:
                answer = chat_with_lifeos(
                    prompt,
                    st.session_state.lifeos_chat_history[:-1]
                )

            except Exception as error:
                answer = (
                    "Sorry, the AI service is temporarily unavailable. "
                    f"Error: {error}"
                )

        st.markdown(answer)

    # Save assistant response
    st.session_state.lifeos_chat_history.append({
        "role": "assistant",
        "content": answer
    })

# ==========================================
# AUTOMATED DAILY PRODUCTIVITY SUMMARY
# ==========================================

st.divider()
st.subheader("📊 Daily Productivity Summary")

if st.button("✨ Generate Today's Summary"):

    with st.spinner("Analyzing your LifeOS data..."):

        try:
            summary = chat_with_lifeos(
                "Generate a concise daily productivity report using my "
                "current LifeOS database. Include pending and overdue "
                "tasks, upcoming deadlines, recent study/gym/volleyball "
                "activities, and recent habit progress. "
                "Give me three practical priorities. "
                "Clearly distinguish missing data from zero activity."
            )

            st.session_state.daily_productivity_summary = summary

        except Exception as error:
            st.error(f"Could not generate summary: {error}")

if "daily_productivity_summary" in st.session_state:
    st.markdown(st.session_state.daily_productivity_summary)
