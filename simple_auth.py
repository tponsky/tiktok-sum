"""
Simple JWT-based authentication using SQLite
For TikTok Video RAG Search
"""
import os
import sqlite3
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Database file - use /app/data/ in Docker, or local directory
DATA_DIR = os.getenv("DATA_DIR", ".")
DB_FILE = os.path.join(DATA_DIR, "users.db")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None


# Database connection
def get_db_connection():
    """Get database connection with auto-migration."""
    db_path = Path(DB_FILE)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Auto-migrate if tables don't exist
    try:
        cur = conn.cursor()

        # Create users table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cur.fetchone():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stripe_customer_id TEXT,
                    balance_usd REAL DEFAULT 0.0
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
            conn.commit()

        # Create usage table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage'")
        if not cur.fetchone():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    details TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);")
            conn.commit()

        # Create password_reset_tokens table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='password_reset_tokens'")
        if not cur.fetchone():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reset_token ON password_reset_tokens(token);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reset_user_id ON password_reset_tokens(user_id);")
            conn.commit()
    except Exception as e:
        print(f"Database migration error: {e}")

    return conn


# Password hashing functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# User database functions
def get_user_by_email(email: str):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        return cur.fetchone()
    finally:
        conn.close()

def get_user_by_id(user_id: int):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()
    finally:
        conn.close()

def create_user(email: str, password: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        hashed_password = get_password_hash(password)

        # Give new users $2.00 trial balance
        cur.execute(
            "INSERT INTO users (email, hashed_password, balance_usd) VALUES (?, ?, ?)",
            (email, hashed_password, 2.00)
        )
        user_id = cur.lastrowid
        conn.commit()

        # Fetch the created user
        cur.execute("SELECT id, email, created_at, balance_usd FROM users WHERE id = ?", (user_id,))
        user = cur.fetchone()
        return user
    except sqlite3.IntegrityError:
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error creating user: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()


# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return TokenData(email=email)
    except JWTError:
        return None


# Authentication functions
def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user['hashed_password']):
        return False
    return user


# ============================================================================
# Usage Tracking
# ============================================================================

def log_usage(user_id: int, action: str, tokens_used: int = 0, cost_usd: float = 0.0, details: str = ""):
    """Log API usage for billing."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO usage (user_id, action, tokens_used, cost_usd, details)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, action, tokens_used, cost_usd, details)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error logging usage: {e}")
    finally:
        conn.close()


def get_user_usage(user_id: int, start_date: str = None, end_date: str = None):
    """Get user's usage records within date range."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if start_date and end_date:
            cur.execute(
                """SELECT * FROM usage
                   WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                   ORDER BY timestamp DESC""",
                (user_id, start_date, end_date)
            )
        else:
            cur.execute(
                "SELECT * FROM usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
                (user_id,)
            )
        return cur.fetchall()
    finally:
        conn.close()


def get_user_total_cost(user_id: int, start_date: str = None, end_date: str = None) -> float:
    """Get total cost for user within date range."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if start_date and end_date:
            cur.execute(
                """SELECT SUM(cost_usd) as total FROM usage
                   WHERE user_id = ? AND timestamp BETWEEN ? AND ?""",
                (user_id, start_date, end_date)
            )
        else:
            cur.execute(
                "SELECT SUM(cost_usd) as total FROM usage WHERE user_id = ?",
                (user_id,)
            )
        row = cur.fetchone()
        return row['total'] if row and row['total'] else 0.0
    finally:
        conn.close()


# ============================================================================
# Balance Management
# ============================================================================

def get_user_balance(user_id: int) -> float:
    """Get user's current balance."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT balance_usd FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row['balance_usd'] if row else 0.0
    finally:
        conn.close()


def add_to_balance(user_id: int, amount_usd: float):
    """Add funds to user's balance (e.g., after Stripe payment)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET balance_usd = balance_usd + ? WHERE id = ?",
            (amount_usd, user_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def deduct_from_balance(user_id: int, amount_usd: float):
    """Deduct cost from user's balance (after API call)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET balance_usd = balance_usd - ? WHERE id = ?",
            (amount_usd, user_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_stripe_customer_id(user_id: int, customer_id: str):
    """Store Stripe customer ID for user."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
            (customer_id, user_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_stripe_customer_id(user_id: int) -> Optional[str]:
    """Get user's Stripe customer ID."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT stripe_customer_id FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row['stripe_customer_id'] if row else None
    finally:
        conn.close()


# ============================================================================
# Password Reset
# ============================================================================

RESET_TOKEN_EXPIRE_HOURS = 1  # Token expires in 1 hour


def create_password_reset_token(email: str) -> Optional[str]:
    """Create a password reset token for the user."""
    user = get_user_by_email(email)
    if not user:
        return None

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Invalidate any existing tokens for this user
        cur.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE user_id = ? AND used = 0",
            (user['id'],)
        )

        # Create new token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)

        cur.execute(
            """INSERT INTO password_reset_tokens (user_id, token, expires_at)
               VALUES (?, ?, ?)""",
            (user['id'], token, expires_at)
        )
        conn.commit()
        return token
    except Exception as e:
        conn.rollback()
        print(f"Error creating reset token: {e}")
        return None
    finally:
        conn.close()


def verify_reset_token(token: str) -> Optional[dict]:
    """Verify a password reset token and return user info if valid."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT prt.*, u.email FROM password_reset_tokens prt
               JOIN users u ON prt.user_id = u.id
               WHERE prt.token = ? AND prt.used = 0""",
            (token,)
        )
        row = cur.fetchone()

        if not row:
            return None

        # Check if expired
        expires_at = datetime.fromisoformat(row['expires_at'])
        if datetime.utcnow() > expires_at:
            return None

        return {
            "user_id": row['user_id'],
            "email": row['email'],
            "token": token
        }
    finally:
        conn.close()


def reset_password(token: str, new_password: str) -> bool:
    """Reset user's password using a valid token."""
    token_data = verify_reset_token(token)
    if not token_data:
        return False

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Update password
        hashed_password = get_password_hash(new_password)
        cur.execute(
            "UPDATE users SET hashed_password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (hashed_password, token_data['user_id'])
        )

        # Mark token as used
        cur.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE token = ?",
            (token,)
        )

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error resetting password: {e}")
        return False
    finally:
        conn.close()


def update_password(user_id: int, new_password: str) -> bool:
    """Update user's password directly (for authenticated password change)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        hashed_password = get_password_hash(new_password)
        cur.execute(
            "UPDATE users SET hashed_password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (hashed_password, user_id)
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error updating password: {e}")
        return False
    finally:
        conn.close()
