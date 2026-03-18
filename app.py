import json
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn.functional as F
import torch.nn as nn
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "smartscreen_ultra_2025"
DB = 'database.db'

SENDER_EMAIL = "triossoftwaremail@gmail.com"
SENDER_PASSWORD = "knaxddlwfpkplsik" 
STATIC_TASKS = {
    "dyslexia": [
        {"id":"dys_1","title":"Letter Mirror Practice","desc":"Write b, d, p, q five times each. Focus on direction. Draw arrows showing which way letters face to build visual memory.","duration":"15 min","level":"Beginner","icon":"🔤","skills":"Letter orientation, visual discrimination","tip":"Use different ink colours for each letter pair (b/d and p/q)."},
        {"id":"dys_2","title":"Phonics Sound Mapping","desc":"Listen to 10 words and write the first sound. Practice sh, ch, th, wh blends with word examples. Make flash cards.","duration":"20 min","level":"Intermediate","icon":"🎵","skills":"Phonemic awareness, sound blending","tip":"Record yourself saying each sound and play it back to check."},
        {"id":"dys_3","title":"Syllable Clapping","desc":"Clap and count syllables for 15 multi-syllable words. Break words into chunks and write them segmented (e.g. but-ter-fly).","duration":"10 min","level":"Beginner","icon":"👏","skills":"Syllabification, word decoding","tip":"Start with 2-syllable words, then build up to 4-syllable ones."},
        {"id":"dys_4","title":"Word Family Builder","desc":"Build 5 word families (cat→bat→mat→rat). Write 4 rhyming words for each base word on separate index cards for revision.","duration":"25 min","level":"Intermediate","icon":"🔨","skills":"Orthographic patterns, rhyme recognition","tip":"Keep all your cards in a box and test yourself daily."},
        {"id":"dys_5","title":"Reading Fluency Sprint","desc":"Read a passage aloud for 2 minutes while running a finger under each line. Count correct words per minute and aim to beat your score.","duration":"30 min","level":"Advanced","icon":"📚","skills":"Reading fluency, automaticity","tip":"A coloured transparent overlay can reduce visual crowding on the page."},
    ],
    "adhd": [
        {"id":"adhd_1","title":"5-4-3-2-1 Grounding","desc":"Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Write each item in a notebook to anchor attention.","duration":"5 min","level":"Beginner","icon":"🧘","skills":"Attention regulation, mindfulness","tip":"Do this before starting any study session to calm your mind."},
        {"id":"adhd_2","title":"Pomodoro Focus Block","desc":"Work on one single task for 15 minutes with zero distractions — phone face-down, notifications off. Take a 3-minute movement break, then repeat.","duration":"18 min","level":"Intermediate","icon":"⏱️","skills":"Sustained attention, impulse control","tip":"A physical sand timer works better than a phone timer."},
        {"id":"adhd_3","title":"Task Sequencing Chart","desc":"Each evening, write tomorrow's top 5 tasks and number them by priority. Draw checkboxes and tick each one when complete. Review at bedtime.","duration":"10 min","level":"Beginner","icon":"📋","skills":"Executive function, planning","tip":"Colour-code tasks by subject or urgency level."},
        {"id":"adhd_4","title":"Stop Signal Practice","desc":"Clap to a steady rhythm. Stop immediately when someone says STOP and restart only on GO. Play 3 rounds of 2 minutes each to build impulse control.","duration":"12 min","level":"Intermediate","icon":"🚦","skills":"Inhibitory control, response monitoring","tip":"Trains the brain's inhibitory system — very effective for ADHD."},
        {"id":"adhd_5","title":"Working Memory Journal","desc":"Read one short paragraph, close the book, then write 5 key facts purely from memory. Compare your notes with the original and highlight gaps.","duration":"20 min","level":"Advanced","icon":"🧠","skills":"Working memory, recall accuracy","tip":"Use this with textbook sections when studying for exams."},
    ]
}

# =============================================================================
#  DATABASE
# =============================================================================

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL,
        password TEXT NOT NULL,
        condition TEXT DEFAULT 'none',
        screened INTEGER DEFAULT 0,
        parent_email TEXT,
        assigned_teacher_id INTEGER,
        disorder_skill TEXT DEFAULT 'both',
        preferred_language TEXT DEFAULT 'en',
        teacher_in_charge INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS teacher_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        teacher_id INTEGER NOT NULL,
        disorder TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id INTEGER NOT NULL,
        receiver_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_read INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS assigned_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        teacher_id INTEGER NOT NULL,
        task_id TEXT NOT NULL,
        disorder TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed_at DATETIME,
        notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS game_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        game_id TEXT NOT NULL,
        score INTEGER DEFAULT 0,
        disorder TEXT NOT NULL,
        played_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    for col, definition in [
        ("disorder_skill",    "TEXT DEFAULT 'both'"),
        ("teacher_in_charge", "INTEGER DEFAULT 0"),
        ("preferred_language","TEXT DEFAULT 'en'"),
    ]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
        except Exception:
            pass
    conn.commit()
    conn.close()

init_db()

# =============================================================================
#  ML MODELS
# =============================================================================

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, padding=0)
        self.fc    = nn.Linear(32 * 2, 16)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

cnn_model = CNN1D()
if os.path.exists("cnn_model.pth"):
    try:
        cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
    except Exception:
        pass
cnn_model.eval()

try:
    svm_model = joblib.load("svm_model.pkl") if os.path.exists("svm_model.pkl") else None
    xgb_model = joblib.load("xgb_model.pkl") if os.path.exists("xgb_model.pkl") else None
    scaler    = joblib.load("scaler.pkl")    if os.path.exists("scaler.pkl")    else None
except Exception:
    svm_model = xgb_model = scaler = None

# =============================================================================
#  SCORING HELPERS
# =============================================================================

def rule_based_predict(raw):
    language, memory, speed, visual, audio, survey = raw
    score = 0
    if language < 4: score += 2
    elif language < 6: score += 1
    if visual < 4: score += 3
    elif visual < 6: score += 1
    if audio < 4: score += 2
    elif audio < 6: score += 1
    if memory < 3: score += 3
    elif memory < 5: score += 1
    if speed < 3: score += 2
    elif speed < 5: score += 1
    if survey < 6: score += 1
    if score >= 6: return 0
    elif score >= 3: return 1
    return 2

def detect_disorder(raw):
    language, memory, speed, visual, audio, survey = raw
    dys = adhd = 0
    if language < 5: dys += 2
    if visual   < 5: dys += 3
    if audio    < 5: dys += 2
    if survey   < 7: dys += 1
    if memory   < 4: adhd += 3
    if speed    < 4: adhd += 2
    if dys > adhd: return "dyslexia"
    if adhd > dys: return "adhd"
    return "both"

def create_teacher_requests(conn, student_id, disorder):
    conn.execute("DELETE FROM teacher_requests WHERE student_id=? AND status='pending'", (student_id,))
    if disorder == "both":
        teachers = conn.execute("SELECT id FROM users WHERE role='teacher'").fetchall()
    else:
        teachers = conn.execute(
            "SELECT id FROM users WHERE role='teacher' AND (disorder_skill=? OR disorder_skill='both')",
            (disorder,)
        ).fetchall()
    for t in teachers:
        conn.execute(
            "INSERT INTO teacher_requests (student_id,teacher_id,disorder,status) VALUES (?,?,?,'pending')",
            (student_id, t['id'], disorder)
        )
    conn.commit()

# =============================================================================
#  EMAIL SYSTEM
#  4 auto-triggers:
#    1. notify_screening_complete  — after student finishes AI screening
#    2. notify_teacher_assigned    — when teacher accepts / admin assigns
#    3. notify_task_assigned       — when teacher assigns a task
#    4. notify_task_completed      — full progress report when student finishes
# =============================================================================

def _send_raw(to_email: str, subject: str, html: str) -> bool:
    try:
        msg = MIMEMultipart('alternative')
        msg["Subject"] = subject
        msg["From"]    = f"SmartScreen Alerts <{SENDER_EMAIL}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html, "html"))

        import ssl
        context = ssl.create_default_context()

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()

        print(f"[EMAIL OK] {subject} -> {to_email}")
        return True

    except Exception as e:
        print("EMAIL ERROR:", e)
        return False
# --- Shared HTML building blocks ---

def _header(h1: str, sub: str = "") -> str:
    return f"""
    <div style="background:linear-gradient(135deg,#1a3c5e 0%,#264653 100%);
                padding:30px 36px;text-align:center;border-bottom:4px solid #E8614A;">
      <div style="font-size:36px;margin-bottom:8px;">🧠</div>
      <h1 style="margin:0;color:#fff;font-family:'Segoe UI',Arial,sans-serif;
                 font-size:21px;font-weight:700;letter-spacing:.4px;">{h1}</h1>
      {"<p style='margin:6px 0 0;color:rgba(255,255,255,.72);font-size:13px;font-family:Segoe UI,Arial,sans-serif;'>"+sub+"</p>" if sub else ""}
    </div>"""

def _footer(year: int) -> str:
    return f"""
    <div style="background:#f8f9fa;padding:22px 36px;text-align:center;
                border-top:1px solid #e5e0d8;font-size:12px;color:#999;
                font-family:'Segoe UI',Arial,sans-serif;">
      <strong style="color:#555;">SmartScreen AI Learning Platform</strong><br>
      Automated notification — do not reply to this email.<br>
      &copy; {year} SmartScreen. All rights reserved.
    </div>"""

def _row(label: str, value: str) -> str:
    return f"""<tr>
      <td style="padding:11px 16px;font-size:13px;color:#777;font-weight:700;width:38%;
                 border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{label}</td>
      <td style="padding:11px 16px;font-size:13px;color:#1a1a2e;
                 border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{value}</td>
    </tr>"""

def _bar(label: str, val: float, color: str = "#2A9D8F") -> str:
    pct = min(100, int(val * 10))
    return f"""
    <div style="margin-bottom:11px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;font-weight:700;color:#555;
                     font-family:'Segoe UI',Arial,sans-serif;">{label}</span>
        <span style="font-size:12px;color:#aaa;
                     font-family:'Segoe UI',Arial,sans-serif;">{val}/10</span>
      </div>
      <div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;"></div>
      </div>
    </div>"""

def _risk_col(r: str) -> str:
    return "#c0392b" if "High" in r else "#e67e22" if "Moderate" in r else "#27ae60"

def _dis_col(d: str) -> str:
    return {"dyslexia":"#E8614A","adhd":"#E9A84C","both":"#7B5EA7"}.get(d, "#264653")

def _dis_label(d: str) -> str:
    return {"dyslexia":"🔤 Dyslexia","adhd":"⚡ ADHD","both":"🔀 Dyslexia + ADHD"}.get(d, d.title())

def _scores_block(details: dict) -> str:
    if not details: return ""
    return f"""
    <div style="background:#f0ede8;border-radius:14px;padding:20px;
                border:1px solid #e5e0d8;margin-bottom:22px;">
      <div style="font-size:14px;font-weight:700;color:#264653;
                  font-family:'Segoe UI',Arial,sans-serif;margin-bottom:14px;">
        📊 Cognitive Skill Scores
      </div>
      {_bar("Language &amp; Vocabulary", details.get("language", 0), "#E8614A")}
      {_bar("Memory &amp; Recall",       details.get("memory",   0), "#7B5EA7")}
      {_bar("Processing Speed",          details.get("speed",    0), "#E9A84C")}
      {_bar("Visual Processing",         details.get("visual",   0), "#2A9D8F")}
      {_bar("Audio Processing",          details.get("audio",    0), "#264653")}
    </div>"""

def _btn(label: str, color: str = "#E8614A") -> str:
    return f"""
    <div style="text-align:center;margin-top:6px;">
      <a href="http://127.0.0.1:2104/student_dashboard"
         style="display:inline-block;padding:13px 32px;
                background:linear-gradient(135deg,{color},{color}cc);
                color:#fff;text-decoration:none;border-radius:100px;
                font-weight:700;font-size:14px;font-family:'Segoe UI',Arial,sans-serif;
                box-shadow:0 4px 14px rgba(0,0,0,.18);">{label}</a>
    </div>"""

# =============================================================================
# TRIGGER 1 — Screening Complete
# =============================================================================
def notify_screening_complete(student_id: int, details: dict):
    conn = get_db()
    row = conn.execute(
        "SELECT name,email,parent_email FROM users WHERE id=?",
        (student_id,)
    ).fetchone()
    conn.close()

    if not row:
    return

name = row['name']

to_email = row['parent_email'] if row['parent_email'] else row['email']

print("SENDING EMAIL TO:", to_email)

   
    risk     = details.get('prediction','Unknown')
    disorder = details.get('disorder','unknown')
    ref      = f"SS-SCR-{uuid.uuid4().hex[:6].upper()}"
    year     = datetime.now().year
    date_str = datetime.now().strftime('%d %B, %Y  %I:%M %p')
    rc       = _risk_col(risk)
    dc       = _dis_col(disorder)
    dl       = _dis_label(disorder)
    icon     = "⚠️" if "High" in risk else "🔶" if "Moderate" in risk else "✅"

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f5f3ee;">
    <div style="max-width:600px;margin:30px auto;border-radius:16px;overflow:hidden;
                box-shadow:0 8px 32px rgba(0,0,0,.12);border:1px solid #e5e0d8;">
      {_header("AI Screening Report", f"Assessment complete for {name}")}
      <div style="background:#fff;padding:32px 36px;">

        <p style="font-size:15px;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;margin:0 0 6px;">
          Dear <strong>Parent / Guardian of {name}</strong>,
        </p>
        <p style="font-size:14px;color:#6b7280;font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;margin:0 0 24px;">
          {name} has just completed the SmartScreen AI-powered learning assessment.
          Here is a full summary of their results. A specialist teacher has been notified
          and will be assigned to your child shortly.
        </p>

        <!-- Risk banner -->
        <div style="background:{rc};color:#fff;padding:18px;border-radius:12px;
                    text-align:center;margin-bottom:20px;">
          <div style="font-size:30px;margin-bottom:6px;">{icon}</div>
          <div style="font-size:19px;font-weight:700;font-family:'Segoe UI',Arial,sans-serif;">
            {risk}
          </div>
          <div style="font-size:12px;opacity:.85;margin-top:4px;
                      font-family:'Segoe UI',Arial,sans-serif;">AI Prediction</div>
        </div>

        <!-- Disorder tag -->
        <div style="text-align:center;margin-bottom:24px;">
          <span style="display:inline-block;padding:8px 22px;border-radius:100px;
                       background:{dc}18;border:1.5px solid {dc}44;
                       font-size:14px;font-weight:700;color:{dc};
                       font-family:'Segoe UI',Arial,sans-serif;">{dl} Indicators Detected</span>
        </div>

        <!-- Details table -->
        <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;
                      border-radius:12px;overflow:hidden;margin-bottom:22px;">
          {_row("Student Name",  name)}
          {_row("Assessment Date", date_str)}
          {_row("Risk Level",   f'<span style="color:{rc};font-weight:700;">{risk}</span>')}
          {_row("Disorder",     f'<span style="color:{dc};font-weight:700;">{dl}</span>')}
          {_row("Reference",    ref)}
        </table>

        <!-- Score bars -->
        {_scores_block(details)}

        <!-- What next -->
        <div style="background:#e8f4f0;border-left:4px solid #2A9D8F;
                    border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:24px;">
          <div style="font-size:13px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:6px;">
            📌 What happens next?
          </div>
          <ul style="margin:0;padding-left:18px;font-size:13px;color:#555;
                     font-family:'Segoe UI',Arial,sans-serif;line-height:1.85;">
            <li>A specialist teacher will review {name}'s results and accept their case.</li>
            <li>You will receive another email once a teacher has been assigned.</li>
            <li>Personalised learning tasks will begin after assignment.</li>
          </ul>
        </div>

        {_btn("View Student Dashboard →", "#E8614A")}
      </div>
      {_footer(year)}
    </div></body></html>"""

    _send_raw(to_email, f"📋 SmartScreen Assessment Complete — {name}", html)

# =============================================================================
# TRIGGER 2 — Teacher Assigned
# =============================================================================
def notify_teacher_assigned(student_id: int, teacher_id: int):
    conn    = get_db()
    student = conn.execute("SELECT name,parent_email,condition FROM users WHERE id=?", (student_id,)).fetchone()
    teacher = conn.execute("SELECT name,email,phone,disorder_skill FROM users WHERE id=?", (teacher_id,)).fetchone()
    t_done  = conn.execute("SELECT COUNT(*) as c FROM assigned_tasks WHERE student_id=? AND status='completed'", (student_id,)).fetchone()['c']
    t_total = conn.execute("SELECT COUNT(*) as c FROM assigned_tasks WHERE student_id=?", (student_id,)).fetchone()['c']
    conn.close()

    if not student or not student['parent_email'] or not teacher:
        return

    name, parent_email = student['name'], student['parent_email']
    details = {}
    try:
        details = json.loads(student['condition']) if student['condition'] else {}
    except Exception:
        pass

    t_name  = teacher['name']
    t_email = teacher['email']
    t_phone = teacher['phone']
    t_skill_raw = teacher['disorder_skill']
    t_skill = {"dyslexia":"🔤 Dyslexia Specialist","adhd":"⚡ ADHD Specialist",
               "both":"🔀 All Disorders"}.get(t_skill_raw, t_skill_raw)
    disorder = details.get('disorder','unknown')
    risk     = details.get('prediction','Unknown')
    dc       = _dis_col(disorder)
    rc       = _risk_col(risk)
    ref      = f"SS-TCH-{uuid.uuid4().hex[:6].upper()}"
    year     = datetime.now().year
    date_str = datetime.now().strftime('%d %B, %Y  %I:%M %p')

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f5f3ee;">
    <div style="max-width:600px;margin:30px auto;border-radius:16px;overflow:hidden;
                box-shadow:0 8px 32px rgba(0,0,0,.12);border:1px solid #e5e0d8;">
      {_header("Teacher Assigned! 🎓", f"A specialist has accepted {name}'s case")}
      <div style="background:#fff;padding:32px 36px;">

        <p style="font-size:15px;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;margin:0 0 6px;">
          Dear <strong>Parent / Guardian of {name}</strong>,
        </p>
        <p style="font-size:14px;color:#6b7280;font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;margin:0 0 24px;">
          Great news! A specialist educator has been matched to {name} and accepted their case.
          Personalised learning tasks will begin shortly.
        </p>

        <!-- Success banner -->
        <div style="background:linear-gradient(135deg,#2A9D8F,#1d7a6e);color:#fff;
                    padding:18px;border-radius:12px;text-align:center;margin-bottom:24px;">
          <div style="font-size:28px;margin-bottom:6px;">🎓</div>
          <div style="font-size:18px;font-weight:700;font-family:'Segoe UI',Arial,sans-serif;">
            Teacher Successfully Assigned
          </div>
          <div style="font-size:12px;opacity:.85;margin-top:4px;
                      font-family:'Segoe UI',Arial,sans-serif;">{date_str}</div>
        </div>

        <!-- Teacher card -->
        <div style="background:#edf7f6;border:1.5px solid #2A9D8F44;
                    border-radius:14px;padding:20px 24px;margin-bottom:24px;">
          <div style="display:flex;gap:14px;align-items:center;margin-bottom:14px;">
            <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;
                        background:linear-gradient(135deg,#2A9D8F,#3BC4B5);
                        display:flex;align-items:center;justify-content:center;
                        font-size:22px;font-weight:700;color:#fff;">
              {t_name[0].upper()}
            </div>
            <div>
              <div style="font-size:16px;font-weight:700;color:#264653;
                          font-family:'Segoe UI',Arial,sans-serif;">{t_name}</div>
              <div style="font-size:13px;color:#2A9D8F;font-weight:600;
                          font-family:'Segoe UI',Arial,sans-serif;">{t_skill}</div>
            </div>
          </div>
          <table style="width:100%;border-collapse:collapse;">
            {_row("Teacher Email", t_email)}
            {_row("Teacher Phone", t_phone)}
            {_row("Specialization", t_skill)}
          </table>
        </div>

        <!-- Student status -->
        <div style="background:#f8f9fa;border-radius:14px;padding:20px;
                    border:1px solid #e5e0d8;margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:12px;">
            📋 Student Status at Assignment
          </div>
          <table style="width:100%;border-collapse:collapse;">
            {_row("AI Risk Level", f'<span style="color:{rc};font-weight:700;">{risk}</span>')}
            {_row("Disorder",      f'<span style="color:{dc};font-weight:700;">{_dis_label(disorder)}</span>')}
            {_row("Tasks Assigned", str(t_total))}
            {_row("Tasks Done",     str(t_done))}
            {_row("Reference",      ref)}
          </table>
        </div>

        {_scores_block(details)}

        <!-- Tip box -->
        <div style="background:#fff8e7;border-left:4px solid #E9A84C;
                    border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:24px;">
          <div style="font-size:13px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:6px;">
            💡 Tips for Parents
          </div>
          <ul style="margin:0;padding-left:18px;font-size:13px;color:#555;
                     font-family:'Segoe UI',Arial,sans-serif;line-height:1.85;">
            <li>Encourage {name} to complete assigned tasks daily.</li>
            <li>Your child can now chat directly with {t_name} on the platform.</li>
            <li>You will receive a progress update each time a task is completed.</li>
          </ul>
        </div>

        {_btn("View Student Dashboard →", "#2A9D8F")}
      </div>
      {_footer(year)}
    </div></body></html>"""

    _send_raw(parent_email, f"🎓 Teacher Assigned to {name} — SmartScreen", html)


# =============================================================================
# TRIGGER 3 — Task Assigned
# =============================================================================
def notify_task_assigned(student_id: int, task: dict, teacher_name: str):
    conn = get_db()
    row     = conn.execute("SELECT name,parent_email FROM users WHERE id=?", (student_id,)).fetchone()
    t_done  = conn.execute("SELECT COUNT(*) as c FROM assigned_tasks WHERE student_id=? AND status='completed'", (student_id,)).fetchone()['c']
    t_total = conn.execute("SELECT COUNT(*) as c FROM assigned_tasks WHERE student_id=?", (student_id,)).fetchone()['c']
    conn.close()

    if not row or not row['parent_email']:
        return

    name, parent_email = row['name'], row['parent_email']
    year     = datetime.now().year
    date_str = datetime.now().strftime('%d %B, %Y  %I:%M %p')
    ref      = f"SS-TSK-{uuid.uuid4().hex[:6].upper()}"
    dis      = task.get('disorder','general')
    dc       = _dis_col(dis)
    pct      = int((t_done / t_total) * 100) if t_total else 0

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f5f3ee;">
    <div style="max-width:600px;margin:30px auto;border-radius:16px;overflow:hidden;
                box-shadow:0 8px 32px rgba(0,0,0,.12);border:1px solid #e5e0d8;">
      {_header("New Task Assigned! 📋", f"{name} has received a personalised learning activity")}
      <div style="background:#fff;padding:32px 36px;">

        <p style="font-size:15px;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;margin:0 0 6px;">
          Dear <strong>Parent / Guardian of {name}</strong>,
        </p>
        <p style="font-size:14px;color:#6b7280;font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;margin:0 0 24px;">
          {teacher_name} has assigned a new learning task to {name}.
          Please encourage your child to complete this activity as part of their learning plan.
        </p>

        <!-- Task card -->
        <div style="background:{dc}10;border:1.5px solid {dc}33;
                    border-radius:14px;padding:22px 24px;margin-bottom:22px;">
          <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;">
            <div style="font-size:2.2rem;line-height:1;">{task.get('icon','📋')}</div>
            <div>
              <div style="font-size:16px;font-weight:700;color:#264653;
                          font-family:'Segoe UI',Arial,sans-serif;">
                {task.get('title','Task')}
              </div>
              <div style="margin-top:6px;">
                <span style="display:inline-block;padding:3px 10px;border-radius:20px;
                             background:{dc}22;color:{dc};font-size:11px;font-weight:700;
                             font-family:'Segoe UI',Arial,sans-serif;">{dis.title()}</span>
                <span style="display:inline-block;padding:3px 10px;border-radius:20px;
                             background:#f0ede8;color:#6b7280;font-size:11px;font-weight:700;
                             font-family:'Segoe UI',Arial,sans-serif;margin-left:5px;">
                  {task.get('level','—')}
                </span>
                <span style="display:inline-block;padding:3px 10px;border-radius:20px;
                             background:#f0ede8;color:#6b7280;font-size:11px;font-weight:700;
                             font-family:'Segoe UI',Arial,sans-serif;margin-left:5px;">
                  ⏱ {task.get('duration','—')}
                </span>
              </div>
            </div>
          </div>
          <p style="font-size:13px;color:#555;font-family:'Segoe UI',Arial,sans-serif;
                    line-height:1.75;margin:0 0 10px;">{task.get('desc','')}</p>
          <div style="font-size:12px;color:#999;font-family:'Segoe UI',Arial,sans-serif;">
            🏷 Skills: <strong style="color:#555;">{task.get('skills','—')}</strong>
          </div>
        </div>

        <!-- Progress bar -->
        <div style="background:#f8f9fa;border-radius:14px;padding:20px;
                    border:1px solid #e5e0d8;margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:10px;">
            📈 Current Progress
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:13px;color:#555;font-family:'Segoe UI',Arial,sans-serif;">
              {t_done} of {t_total} tasks completed
            </span>
            <span style="font-size:13px;font-weight:700;color:#2A9D8F;
                         font-family:'Segoe UI',Arial,sans-serif;">{pct}%</span>
          </div>
          <div style="height:10px;background:#e5e0d8;border-radius:5px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;
                        background:linear-gradient(90deg,#2A9D8F,#3BC4B5);
                        border-radius:5px;"></div>
          </div>
        </div>

        <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;
                      border-radius:12px;overflow:hidden;margin-bottom:24px;">
          {_row("Assigned By",  teacher_name)}
          {_row("Assigned On",  date_str)}
          {_row("Reference",    ref)}
        </table>

        {_btn("View Task in Dashboard →", dc)}
      </div>
      {_footer(year)}
    </div></body></html>"""

    _send_raw(parent_email, f"📋 New Task Assigned to {name} — SmartScreen", html)


# =============================================================================
# TRIGGER 4 — Task Completed / Full Progress Report
# =============================================================================
def notify_task_completed(student_id: int, task_id: str, disorder: str):
    conn     = get_db()
    row      = conn.execute("SELECT name,parent_email,condition FROM users WHERE id=?", (student_id,)).fetchone()
    teacher  = conn.execute(
        "SELECT u.name,u.email FROM users u JOIN users s ON s.assigned_teacher_id=u.id WHERE s.id=?",
        (student_id,)
    ).fetchone()
    tasks_all = conn.execute(
        "SELECT task_id,disorder,status FROM assigned_tasks WHERE student_id=?", (student_id,)
    ).fetchall()
    games_all = conn.execute(
        "SELECT game_id,MAX(score) as best,COUNT(*) as plays FROM game_scores WHERE student_id=? GROUP BY game_id",
        (student_id,)
    ).fetchall()
    conn.close()

    if not row or not row['parent_email']:
        return

    name, parent_email = row['name'], row['parent_email']
    details = {}
    try:
        details = json.loads(row['condition']) if row['condition'] else {}
    except Exception:
        pass

    t_done   = sum(1 for t in tasks_all if t['status'] == 'completed')
    t_total  = len(tasks_all)
    pct      = int((t_done / t_total) * 100) if t_total else 0
    risk     = details.get('prediction', 'Unknown')
    dis      = details.get('disorder', disorder)
    dc       = _dis_col(dis)
    rc       = _risk_col(risk)
    t_name   = teacher['name']  if teacher else "Your Teacher"
    year     = datetime.now().year
    date_str = datetime.now().strftime('%d %B, %Y  %I:%M %p')
    ref      = f"SS-PRG-{uuid.uuid4().hex[:6].upper()}"

    completed_task = next(
        (x for x in STATIC_TASKS.get(disorder, []) if x['id'] == task_id), None
    )

    # Build task history rows
    task_rows = ""
    for t in tasks_all:
        td    = next((x for x in STATIC_TASKS.get(t['disorder'],[]) if x['id'] == t['task_id']), None)
        title = td['title'] if td else t['task_id']
        icon  = td['icon']  if td else "📋"
        badge = ('<span style="color:#27ae60;font-weight:700;">✅ Done</span>'
                 if t['status'] == 'completed'
                 else '<span style="color:#e67e22;font-weight:700;">⏳ Pending</span>')
        task_rows += f"""<tr>
          <td style="padding:10px 14px;font-size:13px;border-bottom:1px solid #f0ede8;
                     font-family:'Segoe UI',Arial,sans-serif;">{icon} {title}</td>
          <td style="padding:10px 14px;font-size:12px;color:#888;border-bottom:1px solid #f0ede8;
                     font-family:'Segoe UI',Arial,sans-serif;">{t['disorder'].title()}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #f0ede8;">{badge}</td>
        </tr>"""

    # Build game score rows
    game_rows = ""
    for g in games_all:
        game_rows += f"""<tr>
          <td style="padding:10px 14px;font-size:13px;border-bottom:1px solid #f0ede8;
                     font-family:'Segoe UI',Arial,sans-serif;">
            🎮 {g['game_id'].replace('_',' ').title()}</td>
          <td style="padding:10px 14px;font-size:13px;font-weight:700;color:#E8614A;
                     border-bottom:1px solid #f0ede8;
                     font-family:'Segoe UI',Arial,sans-serif;">{g['best']}</td>
          <td style="padding:10px 14px;font-size:12px;color:#888;border-bottom:1px solid #f0ede8;
                     font-family:'Segoe UI',Arial,sans-serif;">{g['plays']} sessions</td>
        </tr>"""

    th_style = ("padding:10px 14px;text-align:left;font-size:11px;font-weight:800;"
                "text-transform:uppercase;letter-spacing:.06em;color:#6b7280;"
                "background:#f0ede8;font-family:'Segoe UI',Arial,sans-serif;")

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f5f3ee;">
    <div style="max-width:600px;margin:30px auto;border-radius:16px;overflow:hidden;
                box-shadow:0 8px 32px rgba(0,0,0,.12);border:1px solid #e5e0d8;">
      {_header("Progress Report 📈", f"Task Completed — Full Update for {name}")}
      <div style="background:#fff;padding:32px 36px;">

        <p style="font-size:15px;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;margin:0 0 6px;">
          Dear <strong>Parent / Guardian of {name}</strong>,
        </p>
        <p style="font-size:14px;color:#6b7280;font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;margin:0 0 24px;">
          {name} has just completed a learning task! Below is a full progress report
          covering their screening results, all task history, and game performance.
        </p>

        <!-- Completion banner -->
        <div style="background:linear-gradient(135deg,#27ae60,#1e8449);color:#fff;
                    padding:18px;border-radius:12px;text-align:center;margin-bottom:24px;">
          <div style="font-size:28px;margin-bottom:6px;">🏆</div>
          <div style="font-size:18px;font-weight:700;font-family:'Segoe UI',Arial,sans-serif;">
            Task Completed!
          </div>
          {f'<div style="font-size:14px;opacity:.9;margin-top:4px;font-family:Segoe UI,Arial,sans-serif;">{completed_task["icon"]} {completed_task["title"]}</div>' if completed_task else ""}
        </div>

        <!-- Progress bar (big) -->
        <div style="background:#f8f9fa;border-radius:14px;padding:22px;
                    border:1px solid #e5e0d8;margin-bottom:24px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:10px;">
            📈 Overall Progress — {t_done}/{t_total} Tasks
          </div>
          <div style="height:14px;background:#e5e0d8;border-radius:7px;overflow:hidden;margin-bottom:8px;">
            <div style="height:100%;width:{pct}%;
                        background:linear-gradient(90deg,#2A9D8F,#3BC4B5);border-radius:7px;"></div>
          </div>
          <div style="font-size:24px;font-weight:700;color:#2A9D8F;
                      font-family:'Segoe UI',Arial,sans-serif;text-align:right;">{pct}% Complete</div>
        </div>

        <!-- Screening summary + score bars -->
        <div style="background:#f0ede8;border-radius:14px;padding:20px;
                    border:1px solid #e5e0d8;margin-bottom:24px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:12px;">
            🧠 AI Screening Summary
          </div>
          <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
            {_row("AI Risk Level", f'<span style="color:{rc};font-weight:700;">{risk}</span>')}
            {_row("Disorder",      f'<span style="color:{dc};font-weight:700;">{_dis_label(dis)}</span>')}
            {_row("Assigned Teacher", t_name)}
          </table>
          {_bar("Language &amp; Vocabulary", details.get("language",0), "#E8614A")}
          {_bar("Memory &amp; Recall",       details.get("memory",  0), "#7B5EA7")}
          {_bar("Processing Speed",          details.get("speed",   0), "#E9A84C")}
          {_bar("Visual Processing",         details.get("visual",  0), "#2A9D8F")}
          {_bar("Audio Processing",          details.get("audio",   0), "#264653")}
        </div>

        <!-- Task history table -->
        <div style="margin-bottom:24px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:'Segoe UI',Arial,sans-serif;margin-bottom:12px;">
            📋 Full Task History
          </div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;border-radius:12px;overflow:hidden;">
            <thead><tr>
              <th style="{th_style}">Task</th>
              <th style="{th_style}">Type</th>
              <th style="{th_style}">Status</th>
            </tr></thead>
            <tbody style="background:#fff;">{task_rows}</tbody>
          </table>
        </div>

        <!-- Game scores -->
        {"" if not game_rows else f'''
        <div style="margin-bottom:24px;">
          <div style="font-size:14px;font-weight:700;color:#264653;
                      font-family:Segoe UI,Arial,sans-serif;margin-bottom:12px;">
            🎮 Game Performance
          </div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;border-radius:12px;overflow:hidden;">
            <thead><tr>
              <th style="{th_style}">Game</th>
              <th style="{th_style}">Best Score</th>
              <th style="{th_style}">Sessions</th>
            </tr></thead>
            <tbody style="background:#fff;">{game_rows}</tbody>
          </table>
        </div>'''}

        <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;
                      border-radius:12px;overflow:hidden;margin-bottom:24px;">
          {_row("Report Date",  date_str)}
          {_row("Reference",    ref)}
        </table>

        {_btn("View Full Dashboard →", "#E8614A")}
      </div>
      {_footer(year)}
    </div></body></html>"""

    _send_raw(parent_email, f"🏆 Progress Report: {name} completed a task! — SmartScreen", html)


# =============================================================================
#  AUTH
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth/<role>', methods=['POST'])
def auth(role):
    if role == 'admin':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin':
            session.update({'user_id': 0, 'role': 'admin', 'name': 'Administrator'})
            return redirect(url_for('admin_dashboard'))
        flash("Invalid admin credentials.", "danger")
        return redirect(url_for('index'))
    action   = request.form.get('action')
    email    = request.form.get('email', '')
    password = request.form.get('password', '')
    conn = get_db()
    if action == 'register' and role == 'teacher':
        name  = request.form.get('name')
        phone = request.form.get('phone')
        try:
            conn.execute(
                "INSERT INTO users (role,name,email,phone,password) VALUES (?,?,?,?,?)",
                (role, name, email, phone, generate_password_hash(password))
            )
            conn.commit()
            flash("Registered!", "success")
        except Exception:
            flash("Email already registered.", "danger")
    else:
        row = conn.execute("SELECT * FROM users WHERE email=? AND role=?", (email, role)).fetchone()
        if row and check_password_hash(row['password'], password):
            session.update({'user_id': row['id'], 'role': row['role'], 'name': row['name']})
            conn.close()
            return redirect(url_for('student_dashboard' if role == 'student' else 'teacher_dashboard'))
        flash("Invalid credentials.", "danger")
    conn.close()
    return redirect(url_for('index'))

# =============================================================================
#  SCREENING  ── fires EMAIL TRIGGER 1
# =============================================================================

@app.route('/predict_real_time', methods=['POST'])
def predict_real_time():
    if session.get('role') != 'student':
        return jsonify({"status": "error"}), 403
    data = request.get_json()
    print(data)
    language = float(data.get('language', 0))
    memory   = float(data.get('memory', 0))
    speed    = float(data.get('speed', 0))
    visual   = float(data.get('letter_score', data.get('visual', 0)))
    audio    = float(data.get('reading_score', data.get('audio', 0)))
    survey   = float(data.get('survey', 0))
    raw = [language, memory, speed, visual, audio, survey]
    X    = np.array(raw, dtype=np.float32).reshape(1, -1)
    Xs   = X / 10.0
    if scaler is not None:
        try: Xs = scaler.transform(X)
        except Exception: Xs = X / 10.0
    Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        try: emb = cnn_model(Xt).numpy()
        except Exception: emb = Xs
    pred = None

    if xgb_model is not None:
        try:
            pred = int(xgb_model.predict(emb)[0])
            print("xgboost:", pred)
        except Exception:
            pred = None

    if pred is None and svm_model is not None:
        try:
            pred = int(svm_model.predict(emb)[0])
            print("svm:", pred)
        except Exception:
            pred = None

    if pred is None:
        pred = rule_based_predict(raw)
        print("rule:", pred)
    risk_map = {0:"High Risk", 1:"Moderate Risk", 2:"Low Risk (Normal)"}
    disorder  = detect_disorder(raw)
    print(disorder)
    details   = {
        "prediction": risk_map.get(pred,"Unknown"), "disorder": disorder,
        "language":raw[0],"memory":raw[1],"speed":raw[2],
        "visual":raw[3],"audio":raw[4],"survey":raw[5],
        "letter_score":raw[3],"reading_score":raw[4]
    }
    conn = get_db()
    conn.execute("UPDATE users SET screened=1, condition=? WHERE id=?",
                 (json.dumps(details), session['user_id']))
    conn.commit()
    current = conn.execute("SELECT assigned_teacher_id FROM users WHERE id=?", (session['user_id'],)).fetchone()
    if not current or not current['assigned_teacher_id']:
        create_teacher_requests(conn, session['user_id'], disorder)
    conn.close()

    # ── EMAIL TRIGGER 1 ──────────────────────────────────────────────────────
    try: notify_screening_complete(session['user_id'], details)
    except Exception as e: print(f"[EMAIL] Screening: {e}")

    return jsonify({"status":"success","disorder":disorder,"prediction":risk_map.get(pred,"Unknown")})

# =============================================================================
#  STUDENT DASHBOARD
# =============================================================================

@app.route('/student_dashboard')
def student_dashboard():
    if session.get('role') != 'student':
        return redirect(url_for('index'))
    conn = get_db()
    user = conn.execute(
        "SELECT screened,assigned_teacher_id,preferred_language,condition FROM users WHERE id=?",
        (session['user_id'],)
    ).fetchone()
    if user and user['screened'] == 0:
        conn.close()
        return render_template('screening.html')
    teacher_name = teacher_id = None
    if user and user['assigned_teacher_id']:
        t = conn.execute("SELECT id,name FROM users WHERE id=?", (user['assigned_teacher_id'],)).fetchone()
        if t:
            teacher_name = t['name']
            teacher_id   = t['id']
    disorder  = None
    screening = {}
    if user and user['condition']:
        try:
            screening = json.loads(user['condition'])
            disorder  = screening.get('disorder')
        except Exception:
            pass
    tasks_raw = conn.execute(
        "SELECT id,task_id,disorder,status,notes,assigned_at FROM assigned_tasks WHERE student_id=? ORDER BY assigned_at DESC",
        (session['user_id'],)
    ).fetchall()
    scores = conn.execute(
        "SELECT game_id,MAX(score) as best,COUNT(*) as plays,disorder FROM game_scores WHERE student_id=? GROUP BY game_id",
        (session['user_id'],)
    ).fetchall()
    all_score_history = conn.execute(
        "SELECT game_id,score,disorder,played_at FROM game_scores WHERE student_id=? ORDER BY played_at ASC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    task_list = []
    for t in tasks_raw:
        td = next((x for x in STATIC_TASKS.get(t['disorder'],[]) if x['id'] == t['task_id']), None)
        if td:
            task_list.append({**td,"db_id":t['id'],"status":t['status'],
                               "notes":t['notes'],"assigned_at":t['assigned_at'],
                               "tip":td.get('tip','')})
    return render_template('student_dashboard.html',
                           name=session['name'], screened=True,
                           teacher_name=teacher_name, teacher_id=teacher_id,
                           disorder=disorder, screening=screening,
                           lang=user['preferred_language'] if user else 'en',
                           tasks=task_list, game_scores=[dict(s) for s in scores],
                           all_score_history=[dict(s) for s in all_score_history])

# =============================================================================
#  TEACHER DASHBOARD
# =============================================================================

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if session.get('role') != 'teacher':
        return redirect(url_for('index'))
    conn = get_db()
    teacher_info  = conn.execute("SELECT disorder_skill FROM users WHERE id=?", (session['user_id'],)).fetchone()
    teacher_skill = teacher_info['disorder_skill'] if teacher_info else 'both'
    students_raw  = conn.execute(
        "SELECT id,name,email,condition,screened,teacher_in_charge FROM users WHERE role='student' AND assigned_teacher_id=?",
        (session['user_id'],)
    ).fetchall()
    pending_requests = conn.execute(
        """SELECT tr.id as req_id, tr.disorder, tr.created_at as req_at,
               u.id, u.name, u.email, u.condition, u.screened
           FROM teacher_requests tr JOIN users u ON tr.student_id=u.id
           WHERE tr.teacher_id=? AND tr.status='pending' ORDER BY tr.created_at DESC""",
        (session['user_id'],)
    ).fetchall()
    teachers   = conn.execute("SELECT id,name FROM users WHERE role='teacher' AND id!=?", (session['user_id'],)).fetchall()
    unread_map = {
        r['sender_id']: r['cnt']
        for r in conn.execute(
            "SELECT sender_id,COUNT(*) as cnt FROM chat_messages WHERE receiver_id=? AND is_read=0 GROUP BY sender_id",
            (session['user_id'],)
        ).fetchall()
    }
    students = []
    for s in students_raw:
        details = None
        if s['screened']:
            try: details = json.loads(s['condition'])
            except Exception: details = {"prediction": s['condition']}
        tasks = [dict(t) for t in conn.execute(
            "SELECT id,task_id,disorder,status,notes FROM assigned_tasks WHERE student_id=?", (s['id'],)
        ).fetchall()]
        # Ensure risk_improved is always present as a boolean
        if details and 'risk_improved' not in details:
            details['risk_improved'] = False
        students.append({
            "id":s['id'],"name":s['name'],"email":s['email'],
            "screened":s['screened'],"details":details,
            "unread":unread_map.get(s['id'],0),"tasks":tasks,
            "teacher_in_charge":s['teacher_in_charge'] or 0,
        })
    pending_list = []
    for p in pending_requests:
        details = None
        if p['screened']:
            try: details = json.loads(p['condition'])
            except Exception: details = {}
        pending_list.append({
            "req_id":p['req_id'],"id":p['id'],"name":p['name'],
            "email":p['email'],"screened":p['screened'],
            "disorder":p['disorder'],"req_at":p['req_at'],"details":details,
        })
    conn.close()
    return render_template('teacher_dashboard.html',
                           students=students, pending_students=pending_list,
                           name=session['name'], teacher_skill=teacher_skill,
                           teachers=[dict(t) for t in teachers], all_tasks=STATIC_TASKS)

# =============================================================================
#  TEACHER ACTIONS
# =============================================================================

@app.route('/teacher/accept_student', methods=['POST'])
def accept_student():
    if session.get('role') != 'teacher':
        return jsonify({"status":"error"}), 403
    data       = request.get_json()
    student_id = data.get('student_id')
    conn       = get_db()
    already    = conn.execute("SELECT assigned_teacher_id FROM users WHERE id=?", (student_id,)).fetchone()
    if already and already['assigned_teacher_id']:
        conn.close()
        return jsonify({"status":"already_assigned"})
    conn.execute("UPDATE users SET assigned_teacher_id=? WHERE id=?", (session['user_id'], student_id))
    conn.execute("UPDATE teacher_requests SET status='accepted' WHERE student_id=? AND teacher_id=?",
                 (student_id, session['user_id']))
    conn.execute("UPDATE teacher_requests SET status='closed' WHERE student_id=? AND teacher_id!=? AND status='pending'",
                 (student_id, session['user_id']))
    conn.commit()
    conn.close()

    # ── EMAIL TRIGGER 2 ──────────────────────────────────────────────────────
    try: notify_teacher_assigned(student_id, session['user_id'])
    except Exception as e: print(f"[EMAIL] Teacher accept: {e}")

    return jsonify({"status":"ok"})

@app.route('/teacher/decline_student', methods=['POST'])
def decline_student():
    if session.get('role') != 'teacher':
        return jsonify({"status":"error"}), 403
    data = request.get_json()
    conn = get_db()
    conn.execute("UPDATE teacher_requests SET status='declined' WHERE student_id=? AND teacher_id=?",
                 (data.get('student_id'), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"status":"ok"})

@app.route('/teacher/take_charge', methods=['POST'])
def take_charge():
    if session.get('role') != 'teacher':
        return jsonify({"status":"error"}), 403
    data       = request.get_json()
    student_id = data.get('student_id')
    action     = data.get('action','take')
    conn       = get_db()
    student    = conn.execute(
        "SELECT assigned_teacher_id,teacher_in_charge FROM users WHERE id=?", (student_id,)
    ).fetchone()
    if not student or student['assigned_teacher_id'] != session['user_id']:
        conn.close()
        return jsonify({"status":"not_your_student"}), 403
    in_charge = 1 if action == 'take' else 0
    conn.execute("UPDATE users SET teacher_in_charge=? WHERE id=?", (in_charge, student_id))
    conn.commit()
    conn.close()
    return jsonify({"status":"ok","in_charge":in_charge})

@app.route('/teacher/assign_task', methods=['POST'])
def assign_task():
    if session.get('role') != 'teacher':
        return jsonify({}), 403
    data = request.get_json()
    conn = get_db()
    ex   = conn.execute(
        "SELECT id FROM assigned_tasks WHERE student_id=? AND task_id=? AND status='pending'",
        (data.get('student_id'), data.get('task_id'))
    ).fetchone()
    if ex:
        conn.close()
        return jsonify({"status":"already_assigned"})
    conn.execute(
        "INSERT INTO assigned_tasks (student_id,teacher_id,task_id,disorder,notes) VALUES (?,?,?,?,?)",
        (data.get('student_id'), session['user_id'],
         data.get('task_id'), data.get('disorder'), data.get('notes',''))
    )
    conn.commit()
    task_meta = next(
        (x for x in STATIC_TASKS.get(data.get('disorder',''),[]) if x['id'] == data.get('task_id')),
        {"title":data.get('task_id'),"icon":"📋","desc":"","level":"—","duration":"—","skills":"—"}
    )
    task_meta = dict(task_meta)
    task_meta['disorder'] = data.get('disorder','')
    conn.close()

    # ── EMAIL TRIGGER 3 ──────────────────────────────────────────────────────
    try: notify_task_assigned(data.get('student_id'), task_meta, session['name'])
    except Exception as e: print(f"[EMAIL] Task assigned: {e}")

    return jsonify({"status":"ok"})

@app.route('/teacher/reassign_student', methods=['POST'])
def reassign_student():
    if session.get('role') != 'teacher':
        return redirect(url_for('index'))
    conn = get_db()
    conn.execute(
        "UPDATE users SET assigned_teacher_id=? WHERE id=? AND assigned_teacher_id=?",
        (request.form.get('new_teacher_id'), request.form.get('student_id'), session['user_id'])
    )
    conn.commit()
    conn.close()
    flash("Student reassigned.", "success")
    return redirect(url_for('teacher_dashboard'))

# =============================================================================
#  STUDENT ACTIONS  ── fires EMAIL TRIGGER 4
# =============================================================================

def recalculate_risk(student_id: int, conn) -> dict:
    """
    Recalculate student risk after task completion.
    Logic: each completed task boosts relevant skill scores by 0.3 (capped at 10),
    then re-runs the same prediction pipeline.
    Returns updated details dict (or empty if student not screened).
    """
    user = conn.execute("SELECT condition, screened FROM users WHERE id=?", (student_id,)).fetchone()
    if not user or not user['screened'] or not user['condition']:
        return {}
    try:
        details = json.loads(user['condition'])
    except Exception:
        return {}

    # Count completed tasks per disorder
    completed = conn.execute(
        "SELECT disorder, COUNT(*) as cnt FROM assigned_tasks WHERE student_id=? AND status='completed' GROUP BY disorder",
        (student_id,)
    ).fetchall()
    completed_map = {r['disorder']: r['cnt'] for r in completed}
    total_completed = sum(completed_map.values())

    if total_completed == 0:
        return details

    # Store original risk for comparison
    original_risk = details.get('prediction', '')

    # Boost scores: each completed task improves skills by 0.4 (cap at 10)
    boost_per_task = 0.4
    dyslexia_cnt = completed_map.get('dyslexia', 0)
    adhd_cnt     = completed_map.get('adhd', 0)
    both_cnt     = completed_map.get('both', 0)

    def boost(val, amount):
        return min(10.0, float(val or 0) + amount)

    # Dyslexia tasks boost: language, visual, audio
    d_boost = (dyslexia_cnt + both_cnt) * boost_per_task
    a_boost = (adhd_cnt + both_cnt) * boost_per_task

    details['language'] = boost(details.get('language', 0), d_boost)
    details['visual']   = boost(details.get('visual',   0), d_boost)
    details['audio']    = boost(details.get('audio',    0), d_boost)
    # ADHD tasks boost: memory, speed
    details['memory']   = boost(details.get('memory', 0), a_boost)
    details['speed']    = boost(details.get('speed',  0), a_boost)

    # Re-run prediction with updated scores
    raw = [
        details['language'], details['memory'], details['speed'],
        details['visual'], details['audio'], details.get('survey', 5)
    ]
    X    = np.array(raw, dtype=np.float32).reshape(1, -1)
    Xs   = X / 10.0
    if scaler is not None:
        try: Xs = scaler.transform(X)
        except Exception: Xs = X / 10.0
    Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        try: emb = cnn_model(Xt).numpy()
        except Exception: emb = Xs
    pred = None
    if xgb_model is not None:
        try: pred = int(xgb_model.predict(emb)[0])
        except Exception: pred = None
    if pred is None and svm_model is not None:
        try: pred = int(svm_model.predict(emb)[0])
        except Exception: pred = None
    if pred is None:
        pred = rule_based_predict(raw)

    risk_map = {0: "High Risk", 1: "Moderate Risk", 2: "Low Risk (Normal)"}
    new_risk  = risk_map.get(pred, "Unknown")

    # Determine if risk improved (lower value = better)
    risk_order = {"High Risk": 0, "Moderate Risk": 1, "Low Risk (Normal)": 2}
    old_order  = risk_order.get(original_risk, -1)
    new_order  = risk_order.get(new_risk, -1)
    details['initial_risk']  = original_risk
    details['prediction']    = new_risk
    details['risk_improved'] = (new_order > old_order)

    # Persist updated condition back
    conn.execute("UPDATE users SET condition=? WHERE id=?", (json.dumps(details), student_id))
    conn.commit()
    print(f"[RISK] Student {student_id}: {original_risk} → {new_risk} (improved={details['risk_improved']})")
    return details


@app.route('/student/complete_task', methods=['POST'])
def complete_task():
    if session.get('role') != 'student':
        return jsonify({}), 401
    payload = request.get_json()
    tid     = payload.get('task_id')
    score   = payload.get('score', 0)
    conn    = get_db()
    conn.execute(
        "UPDATE assigned_tasks SET status='completed',completed_at=CURRENT_TIMESTAMP WHERE id=? AND student_id=?",
        (tid, session['user_id'])
    )
    conn.commit()
    task_row = conn.execute(
        "SELECT task_id,disorder FROM assigned_tasks WHERE id=?", (tid,)
    ).fetchone()

    # Save activity score as a game score for tracking
    if task_row and score:
        conn.execute(
            "INSERT INTO game_scores (student_id,game_id,score,disorder) VALUES (?,?,?,?)",
            (session['user_id'], 'task_' + task_row['task_id'], score, task_row['disorder'])
        )
        conn.commit()

    # ── RISK RECALCULATION ───────────────────────────────────────────────────
    updated_details = {}
    try:
        updated_details = recalculate_risk(session['user_id'], conn)
    except Exception as e:
        print(f"[RISK] Recalculation error: {e}")

    conn.close()

    # ── EMAIL TRIGGER 4 ──────────────────────────────────────────────────────
    if task_row:
        try: notify_task_completed(session['user_id'], task_row['task_id'], task_row['disorder'])
        except Exception as e: print(f"[EMAIL] Task completed: {e}")

    return jsonify({
        "status": "ok",
        "score": score,
        "risk_updated": bool(updated_details),
        "new_risk": updated_details.get('prediction', ''),
        "risk_improved": updated_details.get('risk_improved', False),
    })

@app.route('/api/send_report_email/<int:student_id>', methods=['POST'])
def send_report_email(student_id):
    """Generate and send a full progress report to the student's parent email."""
    if session.get('role') not in ['teacher', 'admin']:
        return jsonify({"status": "error"}), 403
    conn = get_db()
    user = conn.execute(
        "SELECT name, email, condition, screened, parent_email, assigned_teacher_id FROM users WHERE id=?",
        (student_id,)
    ).fetchone()
    if not user:
        conn.close()
        return jsonify({"status": "not_found"}), 404
    parent_email = user['parent_email']
    if not parent_email:
        conn.close()
        return jsonify({"status": "no_email"})

    name    = user['name']
    details = {}
    if user['screened'] and user['condition']:
        try: details = json.loads(user['condition'])
        except Exception: pass

    # Task summary
    tasks_raw = conn.execute(
        "SELECT task_id, disorder, status, assigned_at, completed_at FROM assigned_tasks WHERE student_id=? ORDER BY assigned_at DESC",
        (student_id,)
    ).fetchall()
    t_total = len(tasks_raw)
    t_done  = sum(1 for t in tasks_raw if t['status'] == 'completed')
    pct     = round(t_done / t_total * 100) if t_total else 0

    # Game scores
    game_rows_raw = conn.execute(
        "SELECT game_id, MAX(score) as best, COUNT(*) as plays FROM game_scores WHERE student_id=? GROUP BY game_id",
        (student_id,)
    ).fetchall()

    # Teacher name
    t_name = session['name']

    conn.close()

    # Risk info
    risk          = details.get('prediction', 'Not screened')
    initial_risk  = details.get('initial_risk', risk)
    dis           = details.get('disorder', '—')
    risk_improved = details.get('risk_improved', False)
    rc = _risk_col(risk)
    dc = _dis_col(dis)

    year     = datetime.now().year
    date_str = datetime.now().strftime("%d %B %Y")
    ref      = str(uuid.uuid4()).upper()[:12]

    # Build task rows
    task_rows = ""
    for t in tasks_raw:
        td = next((x for x in STATIC_TASKS.get(t['disorder'], []) if x['id'] == t['task_id']), None)
        title  = (td['icon'] + ' ' + td['title']) if td else t['task_id']
        status_color = "#27ae60" if t['status'] == 'completed' else "#e67e22"
        task_rows += f"""<tr>
          <td style="padding:10px 14px;font-size:12px;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{title}</td>
          <td style="padding:10px 14px;font-size:12px;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;color:{status_color};font-weight:700;">{t['disorder'].title()}</td>
          <td style="padding:10px 14px;font-size:12px;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;color:{status_color};font-weight:700;">{t['status'].title()}</td>
        </tr>"""

    game_rows = ""
    for g in game_rows_raw:
        gid = g['game_id'].replace('_', ' ').title()
        game_rows += f"""<tr>
          <td style="padding:10px 14px;font-size:12px;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{gid}</td>
          <td style="padding:10px 14px;font-size:12px;font-weight:700;color:#E8614A;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{g['best']}</td>
          <td style="padding:10px 14px;font-size:12px;color:#888;border-bottom:1px solid #f0ede8;font-family:'Segoe UI',Arial,sans-serif;">{g['plays']} sessions</td>
        </tr>"""

    th_style = ("padding:10px 14px;text-align:left;font-size:11px;font-weight:800;"
                "text-transform:uppercase;letter-spacing:.06em;color:#6b7280;"
                "background:#f0ede8;font-family:'Segoe UI',Arial,sans-serif;")

    # Risk change banner
    risk_banner = ""
    if risk_improved:
        risk_banner = f"""<div style="background:linear-gradient(135deg,#27ae60,#1e8449);color:#fff;padding:14px 18px;border-radius:12px;text-align:center;margin-bottom:18px;">
          <div style="font-size:20px;margin-bottom:4px">🎉</div>
          <div style="font-size:15px;font-weight:700;font-family:'Segoe UI',Arial,sans-serif;">Risk Level Improved!</div>
          <div style="font-size:13px;opacity:.9;margin-top:3px;font-family:'Segoe UI',Arial,sans-serif;">{initial_risk} → <strong>{risk}</strong> after completing tasks</div>
        </div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f5f3ee;">
    <div style="max-width:620px;margin:30px auto;border-radius:16px;overflow:hidden;
                box-shadow:0 8px 32px rgba(0,0,0,.12);border:1px solid #e5e0d8;">
      {_header("Progress Report 📈", f"Full Progress Update for {name}")}
      <div style="background:#fff;padding:32px 36px;">
        <p style="font-size:15px;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;margin:0 0 6px;">
          Dear <strong>Parent / Guardian of {name}</strong>,
        </p>
        <p style="font-size:14px;color:#6b7280;font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;margin:0 0 22px;">
          Below is a comprehensive progress report covering {name}'s screening results, task completion history, risk level update, and game performance.
        </p>

        {risk_banner}

        <!-- Progress bar -->
        <div style="background:#f8f9fa;border-radius:14px;padding:20px;border:1px solid #e5e0d8;margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;font-family:'Segoe UI',Arial,sans-serif;margin-bottom:10px;">
            📈 Overall Progress — {t_done}/{t_total} Tasks Completed
          </div>
          <div style="height:12px;background:#e5e0d8;border-radius:6px;overflow:hidden;margin-bottom:8px;">
            <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#2A9D8F,#3BC4B5);border-radius:6px;"></div>
          </div>
          <div style="font-size:22px;font-weight:700;color:#2A9D8F;font-family:'Segoe UI',Arial,sans-serif;text-align:right;">{pct}% Complete</div>
        </div>

        <!-- AI Screening + Risk -->
        <div style="background:#f0ede8;border-radius:14px;padding:20px;border:1px solid #e5e0d8;margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;font-family:'Segoe UI',Arial,sans-serif;margin-bottom:12px;">🧠 AI Screening Summary</div>
          <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
            {_row("Initial Risk Level", f'<span style="color:{_risk_col(initial_risk)};font-weight:700;">{initial_risk}</span>')}
            {_row("Current Risk Level", f'<span style="color:{rc};font-weight:700;">{risk} {"✅ Improved!" if risk_improved else ""}</span>')}
            {_row("Disorder", f'<span style="color:{dc};font-weight:700;">{_dis_label(dis)}</span>')}
            {_row("Assigned Teacher", t_name)}
          </table>
          {_bar("Language &amp; Vocabulary", details.get("language", 0), "#E8614A")}
          {_bar("Memory &amp; Recall",       details.get("memory",   0), "#7B5EA7")}
          {_bar("Processing Speed",          details.get("speed",    0), "#E9A84C")}
          {_bar("Visual Processing",         details.get("visual",   0), "#2A9D8F")}
          {_bar("Audio Processing",          details.get("audio",    0), "#264653")}
        </div>

        <!-- Task History -->
        <div style="margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;font-family:'Segoe UI',Arial,sans-serif;margin-bottom:12px;">📋 Full Task History</div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;border-radius:12px;overflow:hidden;">
            <thead><tr>
              <th style="{th_style}">Task</th>
              <th style="{th_style}">Type</th>
              <th style="{th_style}">Status</th>
            </tr></thead>
            <tbody style="background:#fff;">{task_rows or '<tr><td colspan="3" style="padding:16px;text-align:center;color:#888;font-family:Segoe UI,Arial,sans-serif;font-size:12px;">No tasks assigned yet.</td></tr>'}</tbody>
          </table>
        </div>

        <!-- Game Scores -->
        {"" if not game_rows else f'''
        <div style="margin-bottom:22px;">
          <div style="font-size:14px;font-weight:700;color:#264653;font-family:Segoe UI,Arial,sans-serif;margin-bottom:12px;">🎮 Game Performance</div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;border-radius:12px;overflow:hidden;">
            <thead><tr>
              <th style="{th_style}">Game</th>
              <th style="{th_style}">Best Score</th>
              <th style="{th_style}">Sessions</th>
            </tr></thead>
            <tbody style="background:#fff;">{game_rows}</tbody>
          </table>
        </div>'''}

        <table style="width:100%;border-collapse:collapse;border:1px solid #e5e0d8;border-radius:12px;overflow:hidden;margin-bottom:22px;">
          {_row("Report Date", date_str)}
          {_row("Report Reference", ref)}
          {_row("Generated By", f"SmartScreen Platform — {t_name}")}
        </table>

        {_btn("View Full Dashboard →", "#2A9D8F")}
      </div>
      {_footer(year)}
    </div></body></html>"""

    ok = _send_raw(parent_email, f"📊 Progress Report for {name} — SmartScreen", html)
    return jsonify({"status": "ok" if ok else "email_failed"})




@app.route('/api/chat/send', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({}), 401
    d = request.get_json()
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_messages (sender_id,receiver_id,message) VALUES (?,?,?)",
        (session['user_id'], d.get('receiver_id'), d.get('message','').strip())
    )
    conn.commit()
    conn.close()
    return jsonify({"status":"sent","timestamp":datetime.now().strftime("%H:%M")})

@app.route('/api/chat/messages/<int:other_id>')
def get_messages(other_id):
    if 'user_id' not in session:
        return jsonify([])
    conn = get_db()
    msgs = conn.execute(
        """SELECT id,sender_id,message,timestamp FROM chat_messages
           WHERE (sender_id=? AND receiver_id=?) OR (sender_id=? AND receiver_id=?)
           ORDER BY timestamp ASC""",
        (session['user_id'], other_id, other_id, session['user_id'])
    ).fetchall()
    conn.execute("UPDATE chat_messages SET is_read=1 WHERE receiver_id=? AND sender_id=?",
                 (session['user_id'], other_id))
    conn.commit()
    conn.close()
    return jsonify([dict(m) for m in msgs])

@app.route('/api/chat/teacher_info')
def get_teacher_info():
    if session.get('role') != 'student':
        return jsonify({})
    conn  = get_db()
    user  = conn.execute("SELECT assigned_teacher_id FROM users WHERE id=?", (session['user_id'],)).fetchone()
    if not user or not user['assigned_teacher_id']:
        conn.close()
        return jsonify({"teacher":None})
    t = conn.execute("SELECT id,name FROM users WHERE id=?", (user['assigned_teacher_id'],)).fetchone()
    conn.close()
    return jsonify({"teacher":dict(t) if t else None})

# =============================================================================
#  MISC API
# =============================================================================

@app.route('/api/set_language', methods=['POST'])
def set_language():
    if 'user_id' not in session:
        return jsonify({}), 401
    conn = get_db()
    conn.execute("UPDATE users SET preferred_language=? WHERE id=?",
                 (request.get_json().get('lang','en'), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"status":"ok"})

@app.route('/api/save_score', methods=['POST'])
def save_score():
    if session.get('role') != 'student':
        return jsonify({}), 401
    d = request.get_json()
    conn = get_db()
    conn.execute(
        "INSERT INTO game_scores (student_id,game_id,score,disorder) VALUES (?,?,?,?)",
        (session['user_id'], d.get('game_id'), d.get('score',0), d.get('disorder','dyslexia'))
    )
    conn.commit()
    conn.close()
    return jsonify({"status":"ok"})

@app.route('/api/student_performance/<int:student_id>')
def student_performance(student_id):
    if session.get('role') not in ['teacher', 'admin']:
        return jsonify({}), 403
    conn   = get_db()
    scores = conn.execute(
        "SELECT game_id,score,disorder,played_at FROM game_scores WHERE student_id=? ORDER BY played_at ASC",
        (student_id,)
    ).fetchall()
    tasks  = conn.execute(
        "SELECT disorder,status,COUNT(*) as cnt FROM assigned_tasks WHERE student_id=? GROUP BY disorder,status",
        (student_id,)
    ).fetchall()
    user   = conn.execute(
        "SELECT name,condition,screened,parent_email FROM users WHERE id=?", (student_id,)
    ).fetchone()
    conn.close()
    screening = {}
    if user and user['screened']:
        try: screening = json.loads(user['condition'])
        except Exception: pass
    return jsonify({
        "scores":       [dict(s) for s in scores],
        "tasks":        [dict(t) for t in tasks],
        "screening":    screening,
        "name":         user['name'] if user else "",
        "parent_email": user['parent_email'] if user else None,
    })

# =============================================================================
#  ADMIN DASHBOARD & ROUTES
# =============================================================================

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    conn = get_db()
    students_raw = conn.execute(
        """SELECT u.id,u.name,u.email,u.phone,u.screened,u.condition,u.created_at,u.teacher_in_charge,
               t.name as teacher_name
           FROM users u LEFT JOIN users t ON u.assigned_teacher_id=t.id
           WHERE u.role='student' ORDER BY u.created_at DESC"""
    ).fetchall()
    teachers  = conn.execute(
        "SELECT id,name,email,phone,disorder_skill,created_at FROM users WHERE role='teacher' ORDER BY created_at DESC"
    ).fetchall()
    all_scores = conn.execute(
        "SELECT student_id,game_id,score,disorder,played_at FROM game_scores ORDER BY played_at ASC"
    ).fetchall()
    all_tasks  = conn.execute(
        "SELECT student_id,disorder,status,assigned_at FROM assigned_tasks ORDER BY assigned_at ASC"
    ).fetchall()
    stats = {
        "total_students":  len(students_raw),
        "total_teachers":  len(teachers),
        "screened":        sum(1 for s in students_raw if s['screened']),
        "total_tasks":     conn.execute("SELECT COUNT(*) as c FROM assigned_tasks").fetchone()['c'],
        "total_games":     conn.execute("SELECT COUNT(*) as c FROM game_scores").fetchone()['c'],
        "completed_tasks": conn.execute("SELECT COUNT(*) as c FROM assigned_tasks WHERE status='completed'").fetchone()['c'],
    }
    students_list = []
    for s in students_raw:
        details = {}
        if s['screened']:
            try: details = json.loads(s['condition'])
            except Exception: pass
        students_list.append({**dict(s),"details":details,"teacher_in_charge":s['teacher_in_charge'] or 0})
    conn.close()
    return render_template('admin_dashboard.html',
                           students=students_list, teachers=[dict(t) for t in teachers],
                           stats=stats, all_scores=[dict(s) for s in all_scores],
                           all_tasks=[dict(t) for t in all_tasks])

@app.route('/admin/add_student', methods=['POST'])
def admin_add_student():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    name, email, phone, parent_email, password, teacher_id = (
        request.form.get(k) for k in ['name','email','phone','parent_email','password','teacher_id']
    )
    assigned = int(teacher_id) if teacher_id and teacher_id.isdigit() else None
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (role,name,email,phone,password,parent_email,screened,assigned_teacher_id) VALUES ('student',?,?,?,?,?,0,?)",
            (name, email, phone, generate_password_hash(password), parent_email, assigned)
        )
        conn.commit()
        flash(f"Student {name} added!", "success")
    except sqlite3.IntegrityError:
        flash("Email already registered.", "danger")
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/assign_teacher', methods=['POST'])
def admin_assign_teacher():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    student_id = request.form.get('student_id')
    teacher_id = request.form.get('teacher_id')
    conn       = get_db()
    student    = conn.execute(
        "SELECT teacher_in_charge FROM users WHERE id=?", (student_id,)
    ).fetchone()
    if student and student['teacher_in_charge']:
        flash("Cannot reassign — teacher has already taken charge.", "danger")
        conn.close()
        return redirect(url_for('admin_dashboard'))
    assigned = int(teacher_id) if teacher_id and teacher_id.isdigit() else None
    conn.execute("UPDATE users SET assigned_teacher_id=? WHERE id=?", (assigned, student_id))
    conn.commit()
    conn.close()
    flash("Teacher assigned successfully.", "success")
    if assigned:
        try: notify_teacher_assigned(int(student_id), assigned)
        except Exception as e: print(f"[EMAIL] Admin assign: {e}")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_teacher', methods=['POST'])
def admin_add_teacher():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    name, email, phone, password = (request.form.get(k) for k in ['name','email','phone','password'])
    disorder_skill = request.form.get('disorder_skill','both')
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (role,name,email,phone,password,disorder_skill) VALUES ('teacher',?,?,?,?,?)",
            (name, email, phone, generate_password_hash(password), disorder_skill)
        )
        conn.commit()
        flash(f"Teacher {name} added!", "success")
    except Exception:
        flash("Email already exists.", "danger")
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_teacher/<int:tid>', methods=['POST'])
def admin_delete_teacher(tid):
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    conn = get_db()
    conn.execute("UPDATE users SET assigned_teacher_id=NULL WHERE assigned_teacher_id=?", (tid,))
    conn.execute("DELETE FROM users WHERE id=? AND role='teacher'", (tid,))
    conn.commit()
    conn.close()
    flash("Teacher removed.", "success")
    return redirect(url_for('admin_dashboard'))

# =============================================================================
#  GAMES / MISC
# =============================================================================

@app.route('/games')
def games():
    if session.get('role') != 'student':
        return redirect(url_for('index'))
    return render_template('games.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


