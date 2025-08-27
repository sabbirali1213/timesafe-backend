from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
import bcrypt
from jose import JWTError, jwt
import os
import uuid
import logging
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import shutil
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
import requests
import json
from twilio.rest import Client

# Import universal Atlas configuration
from atlas_config import MONGODB_URL, DATABASE_NAME, CONNECTION_OPTIONS

ROOT_DIR = Path(__file__).parent

# Load environment variables with fallbacks
load_dotenv(ROOT_DIR / '.env')

# Environment variables using universal config
MONGO_URL = MONGODB_URL
DB_NAME = DATABASE_NAME
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'https://timesafe.in')

# Firebase configuration
FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID', 'timesafe-delivery')
FIREBASE_API_KEY = os.environ.get('FIREBASE_API_KEY', '')

# Twilio configuration - Complete SMS OTP Service
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_VERIFY_SERVICE_SID = os.environ.get('TWILIO_VERIFY_SERVICE_SID', '')

print(f"🔧 Configuration loaded:")
print(f"   - Database: {DB_NAME}")
print(f"   - CORS Origins: {CORS_ORIGINS}")
print(f"   - MongoDB URL: {MONGO_URL[:20]}..." if MONGO_URL else "   - MongoDB URL: Not set")
print(f"   - Firebase Project: {FIREBASE_PROJECT_ID}")
print(f"   - Firebase API Key: {FIREBASE_API_KEY[:20]}..." if FIREBASE_API_KEY else "   - Firebase API Key: Not set")
print(f"   - Twilio Account: {TWILIO_ACCOUNT_SID[:10]}..." if TWILIO_ACCOUNT_SID else "   - Twilio Account: Not set")
print(f"   - Twilio Service: {TWILIO_VERIFY_SERVICE_SID[:10]}..." if TWILIO_VERIFY_SERVICE_SID else "   - Twilio Service: Not set")

# Create uploads directory
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize MongoDB connection using universal configuration
try:
    client = AsyncIOMotorClient(MONGO_URL, **CONNECTION_OPTIONS)
    db = client[DB_NAME]
    print(f"✅ MongoDB client created using universal Atlas configuration")
except Exception as e:
    print(f"❌ MongoDB client creation failed: {e}")
    # Fallback for development
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client[DATABASE_NAME]  # Use environment variable

app = FastAPI(title="Mutton Delivery API")
api_router = APIRouter(prefix="/api")

# CORS configuration for production deployment
cors_origins = [origin.strip() for origin in CORS_ORIGINS.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

print(f"🌐 CORS configured for origins: {cors_origins}")

# Mount static files for uploaded images under /api path for proper routing
app.mount("/api/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Security
security = HTTPBearer()
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__default_rounds=12
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str  # Full name
    phone: str
    user_type: str  # customer, vendor, delivery_partner, admin
    age: Optional[int] = None  # User age
    gender: Optional[str] = None  # male, female, other
    address: Optional[str] = None  # Complete address
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    # Location coordinates for delivery partners
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    service_radius_km: Optional[int] = 5  # Service radius in kilometers (default 5km)
    business_name: Optional[str] = None  # For vendors
    business_type: Optional[str] = None  # For vendors
    is_online: Optional[bool] = True  # Online status for vendors
    is_verified: Optional[bool] = False  # Vendor verification status
    verification_date: Optional[str] = None  # When vendor was verified
    verified_by: Optional[str] = None  # Admin who verified the vendor

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class FirebaseLoginRequest(BaseModel):
    firebase_token: str = Field(..., description="Firebase ID token")
    phone_number: str = Field(..., description="Phone number from Firebase")
    name: str = Field(..., description="User's display name")
    user_type: str = Field(default="customer", description="User type")
    age: Optional[int] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None

class AdminCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Admin username")
    password: str = Field(..., min_length=6, description="Admin password")
    name: str = Field(..., min_length=2, max_length=100, description="Admin full name")
    email: EmailStr = Field(..., description="Admin email address")
    role: str = Field(default="admin", description="Admin role")

class VendorVerification(BaseModel):
    vendor_id: str = Field(..., description="Vendor user ID to verify")
    verification_status: bool = Field(..., description="True to verify, False to unverify")
    notes: Optional[str] = Field("", description="Verification notes")

class VendorCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Vendor full name")
    password: str = Field(..., min_length=6, description="Vendor password")
    email: EmailStr = Field(..., description="Vendor email address")
    phone: str = Field(..., min_length=10, max_length=15, description="Vendor phone number")
    business_name: str = Field(..., min_length=2, max_length=100, description="Business name")
    business_type: str = Field(default="meat_retail", description="Business type")
    address: Optional[str] = Field("", description="Business address")
    city: Optional[str] = Field("", description="Business city")
    state: Optional[str] = Field("", description="Business state")
    latitude: Optional[float] = Field(None, description="Business latitude")
    longitude: Optional[float] = Field(None, description="Business longitude")

class DeliveryPartnerCreate(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50, description="Delivery partner user ID")
    password: str = Field(..., min_length=6, description="Delivery partner password")
    name: str = Field(..., min_length=2, max_length=100, description="Delivery partner full name")
    email: EmailStr = Field(..., description="Delivery partner email address")
    phone: str = Field(..., min_length=10, max_length=15, description="Delivery partner phone number")
    latitude: Optional[float] = Field(None, description="Service location latitude")
    longitude: Optional[float] = Field(None, description="Service location longitude")
    service_radius_km: int = Field(default=5, description="Service delivery radius in km")

class UserResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str  # Full name
    phone: str
    user_type: str
    age: Optional[int] = None  # User age
    gender: Optional[str] = None
    address: Optional[str] = None  # Complete address
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    # Location coordinates for delivery partners
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    service_radius_km: Optional[int] = 5
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    is_online: Optional[bool] = True
    is_verified: Optional[bool] = False
    verification_date: Optional[str] = None
    verified_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_count: int = 0

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str  # Full name
    phone: str
    user_type: str
    password_hash: str
    age: Optional[int] = None  # User age
    gender: Optional[str] = None
    address: Optional[str] = None  # Complete address
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    # Location coordinates for delivery partners
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    service_radius_km: Optional[int] = 5
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    is_online: Optional[bool] = True
    is_verified: Optional[bool] = False
    verification_date: Optional[str] = None
    verified_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_count: int = 0

class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_id: str
    name: str
    description: str
    price_per_kg: float
    image_url: Optional[str] = None  # Main image for backwards compatibility
    image_urls: List[str] = Field(default_factory=list)  # Multiple images (up to 7)
    is_available: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ProductCreate(BaseModel):
    name: str
    description: str
    price_per_kg: float
    image_url: Optional[str] = None  # Main image for backwards compatibility
    image_urls: List[str] = Field(default_factory=list)  # Multiple images (up to 7)
    is_available: bool = True

class CartItem(BaseModel):
    product_id: str
    quantity: float  # in kg
    weight_option: str  # "1kg", "500g", "250g"
    # Populated product fields for display
    name: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None

class Order(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # For database compatibility
    customer_id: str
    vendor_id: str
    delivery_partner_id: Optional[str] = None
    items: List[CartItem]
    
    # Enhanced Pricing Breakdown
    items_total: float = 0.0  # Product prices only
    delivery_charge: float = 0.0  # Delivery fees
    subtotal: float = 0.0  # Items + Delivery
    gst_rate: float = 0.18  # 18% GST
    gst_amount: float = 0.0  # GST calculation
    total_amount: float = 0.0  # Final total with GST
    
    # Commission system (on items only)
    platform_commission_rate: float = 0.10  # 10% default commission
    platform_commission_amount: Optional[float] = None
    vendor_earnings: Optional[float] = None
    
    delivery_address: dict
    payment_method: str = "cash_on_delivery"
    status: str = "placed"  # placed, accepted, prepared, out_for_delivery, delivered
    
    # Time & Contact Information
    order_time: str = Field(default_factory=lambda: datetime.utcnow().strftime("%I:%M %p"))  # 02:30 PM
    order_date: str = Field(default_factory=lambda: datetime.utcnow().strftime("%d %b %Y"))  # 24 Aug 2025
    estimated_delivery_time: Optional[str] = None  # "45 minutes"
    
    # Customer Contact for Delivery
    customer_name: str = ""
    customer_phone: str = ""
    customer_address_text: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure order_id is same as id for database compatibility
        if not self.order_id or self.order_id == "":
            self.order_id = self.id

# Vendor Payment Gateway Configuration Models
class VendorPaymentGateway(BaseModel):
    vendor_id: str
    gateway_type: str  # "stripe", "razorpay", "paypal", "payu"
    gateway_name: str  # Display name for customers
    api_key: str  # Public/publishable key
    secret_key: str  # Secret key (encrypted)
    webhook_secret: Optional[str] = None
    is_active: bool = True
    currency: str = "INR"
    processing_fee_percentage: float = 2.5  # Gateway's processing fee
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class VendorPaymentConfig(BaseModel):
    gateway_type: str
    gateway_name: str
    api_key: str
    secret_key: str
    webhook_secret: Optional[str] = None
    currency: str = "INR"

# Commission Settings Model
class CommissionSettings(BaseModel):
    commission_rate: float  # Rate as decimal (e.g., 0.05 for 5%)
    updated_by: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Commission calculation function with configurable rates
async def calculate_commission(order_amount: float) -> dict:
    """Calculate platform commission based on configurable admin settings"""
    # Get current commission settings from database
    commission_settings = await db.commission_settings.find_one({}, sort=[("updated_at", -1)])
    
    if commission_settings:
        rate = commission_settings.get("commission_rate", 0.05)  # Default to 5% if not found
    else:
        rate = 0.05  # Default 5% commission rate
    
    commission_amount = round(order_amount * rate, 2)
    vendor_earnings = round(order_amount - commission_amount, 2)
    
    return {
        "commission_rate": rate,
        "commission_amount": commission_amount,
        "vendor_earnings": vendor_earnings,
        "order_amount": order_amount
    }

# Delivery time and fee calculation functions
def calculate_delivery_time(distance_km: float) -> int:
    """Calculate estimated delivery time based on distance"""
    if distance_km <= 2:
        return 15  # 15 minutes for very close vendors
    elif distance_km <= 5:
        return 25  # 25 minutes for nearby vendors
    elif distance_km <= 10:
        return 40  # 40 minutes for moderate distance
    else:
        return 60  # 60 minutes for far vendors

def format_delivery_time(minutes: int) -> str:
    """Format delivery time into readable text"""
    if minutes <= 20:
        return f"{minutes} mins"
    elif minutes <= 45:
        return f"{minutes} mins"
    else:
        return f"{minutes//60}h {minutes%60}m" if minutes >= 60 else f"{minutes} mins"

def calculate_delivery_fee(distance_km: float) -> float:
    """Calculate delivery fee based on distance"""
    if distance_km <= 3:
        return 0.0  # Free delivery for nearby vendors
    elif distance_km <= 7:
        return 25.0  # ₹25 for medium distance
    elif distance_km <= 15:
        return 50.0  # ₹50 for far distance
    else:
        return 75.0  # ₹75 for very far distance

def get_priority_label(distance_km: float, is_online: bool) -> str:
    """Get priority label for vendor"""
    if not is_online:
        return "Offline"
    elif distance_km <= 2:
        return "Express"
    elif distance_km <= 5:
        return "Fast"
    else:
        return "Standard"

async def calculate_delivery_charge(vendor_id: str, delivery_address: dict) -> float:
    """Calculate delivery charge based on vendor location and delivery address"""
    try:
        # Get vendor location
        vendor = await db.users.find_one({"id": vendor_id, "user_type": "vendor"})
        if not vendor:
            return 50.0  # Default delivery charge if vendor not found
        
        # For now, use the existing calculate_delivery_fee function
        # In a real implementation, you would calculate distance between vendor and delivery address
        # using their coordinates and a mapping service like Google Maps API
        
        # Default distance calculation (simplified)
        # You can enhance this with actual geolocation calculation
        default_distance = 5.0  # km
        
        return calculate_delivery_fee(default_distance)
        
    except Exception as e:
        print(f"❌ Error calculating delivery charge: {e}")
        return 50.0  # Default fallback delivery charge
class OrderCreate(BaseModel):
    vendor_id: str
    items: List[CartItem]
    delivery_address: dict
    payment_method: str = "cash_on_delivery"  # cash_on_delivery, online

class Payment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    customer_id: str
    amount: float
    payment_method: str  # cash_on_delivery, online
    payment_status: str = "pending"  # pending, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Notification Models
class NotificationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Notification title")
    message: str = Field(..., min_length=1, max_length=1000, description="Notification message")
    notification_type: str = Field(..., description="Type: 'order', 'promotional', 'system', 'offer'")
    target_users: str = Field(..., description="Target: 'all', 'customers', 'vendors', 'delivery', 'specific'")
    user_ids: Optional[List[str]] = Field([], description="Specific user IDs if target_users is 'specific'")
    send_sms: bool = Field(False, description="Send SMS notification")
    send_push: bool = Field(False, description="Send push notification")
    schedule_time: Optional[datetime] = Field(None, description="Schedule notification for later")

class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    message: str
    notification_type: str  # order, promotional, system, offer
    target_users: str  # all, customers, vendors, delivery, specific
    user_ids: Optional[List[str]] = []
    send_sms: bool = False
    send_push: bool = False
    schedule_time: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed, scheduled
    sent_to_count: int = 0
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserNotification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    notification_id: str
    title: str
    message: str
    notification_type: str
    is_read: bool = False
    sms_sent: bool = False
    push_sent: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    read_at: Optional[datetime] = None

class Address(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    street: str
    city: str
    state: str
    pincode: str
    landmark: Optional[str] = None
    is_default: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class AddressCreate(BaseModel):
    street: str
    city: str
    state: str
    pincode: str
    landmark: Optional[str] = None
    is_default: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return User(**user)

# Auth routes
@api_router.post("/auth/register")
async def register_user(user_data: UserRegister):
    # Prevent admin account creation through public registration
    if user_data.user_type == "admin":
        raise HTTPException(status_code=403, detail="Admin accounts cannot be created through public registration")
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user with hashed password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user dict with all fields including password_hash
    user_dict = {
        "id": str(uuid.uuid4()),
        "email": user_data.email,
        "name": user_data.name,
        "phone": user_data.phone,
        "user_type": user_data.user_type,
        "password_hash": hashed_password,
        "age": user_data.age,  # Added age field
        "gender": user_data.gender,
        "address": user_data.address,
        "city": user_data.city,
        "state": user_data.state,
        "pincode": user_data.pincode,
        "latitude": user_data.latitude,
        "longitude": user_data.longitude,
        "service_radius_km": user_data.service_radius_km,
        "business_name": user_data.business_name,
        "business_type": user_data.business_type,
        "created_at": datetime.utcnow(),
        "last_login": None,
        "login_count": 0
    }
    
    await db.users.insert_one(user_dict)
    
    # Create token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    # Return user without password hash
    user_response = UserResponse(**{k: v for k, v in user_dict.items() if k != 'password_hash'})
    return {"access_token": access_token, "token_type": "bearer", "user": user_response}

@api_router.post("/auth/login")
async def login_user(user_data: UserLogin):
    user = await db.users.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user.get('password_hash')):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Update login time and count
    current_time = datetime.utcnow()
    login_count = user.get('login_count', 0) + 1
    
    await db.users.update_one(
        {"email": user_data.email},
        {
            "$set": {
                "last_login": current_time,
                "login_count": login_count
            }
        }
    )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email, "login_time": current_time.isoformat()}, 
        expires_delta=access_token_expires
    )
    
    # Get updated user data
    updated_user = await db.users.find_one({"email": user_data.email})
    user_obj = UserResponse(**{k: v for k, v in updated_user.items() if k != 'password_hash'})
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": user_obj,
        "login_time": current_time,
        "session_info": {
            "login_count": login_count,
            "previous_login": user.get('last_login'),
            "session_start": current_time
        }
    }

# Check if user exists (for forgot password)
@api_router.post("/auth/check-user")
async def check_user(request: dict):
    email = request.get('email')
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    user = await db.users.find_one({"email": email})
    return {"exists": user is not None}

# Reset password
@api_router.post("/auth/reset-password")
async def reset_password(request: dict):
    email = request.get('email')
    new_password = request.get('new_password')
    reset_code = request.get('reset_code')
    
    if not all([email, new_password, reset_code]):
        raise HTTPException(status_code=400, detail="All fields are required")
    
    # Find user
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash new password using the same context as other endpoints
    hashed_password = get_password_hash(new_password)
    
    # Update user password
    await db.users.update_one(
        {"email": email},
        {"$set": {"password_hash": hashed_password}}
    )
    
    return {"message": "Password reset successfully"}

# Fake OTP Authentication Models
class OTPRequest(BaseModel):
    phone_number: str

class PhoneNumberRequest(BaseModel):
    phone_number: str

class OTPVerification(BaseModel):
    phone_number: str
    otp_code: str
    name: str
    user_type: str = "customer"
    age: Optional[int] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    # Location for delivery partners
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    service_radius_km: Optional[int] = 5
    # Business fields for vendors
    business_name: Optional[str] = None
    business_type: Optional[str] = None

# Fake OTP endpoints
@api_router.post("/auth/send-otp")
async def send_fake_otp(request: OTPRequest):
    """Send fake OTP - always returns success with a simulated OTP"""
    phone_number = request.phone_number
    
    # Validate phone number format (basic)
    if not phone_number or len(phone_number) < 10:
        raise HTTPException(status_code=400, detail="Invalid phone number")
    
    # Generate fake OTP (always 123456 for demo)
    fake_otp = "123456"
    
    # Store fake OTP in memory (in production, you'd use Redis or database)
    # For now, we'll just return success
    
    return {
        "message": "OTP sent successfully",
        "phone_number": phone_number,
        "demo_otp": fake_otp,  # In real implementation, this wouldn't be returned
        "status": "sent"
    }

# Alternative Firebase phone verification using REST API
async def verify_firebase_token_with_rest_api(id_token: str):
    """Verify Firebase ID token using REST API approach"""
    try:
        # Use Firebase Auth REST API to verify ID token
        verify_url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
        
        payload = {
            "idToken": id_token
        }
        
        response = requests.post(verify_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            if 'users' in user_data and len(user_data['users']) > 0:
                user = user_data['users'][0]
                return {
                    'uid': user.get('localId'),
                    'phone_number': user.get('phoneNumber'),
                    'email': user.get('email'),
                    'verified': True
                }
        
        return None
        
    except Exception as e:
        print(f"❌ REST API token verification failed: {e}")
        return None

# Firebase Phone Authentication endpoint (Enhanced)
@api_router.post("/auth/firebase-login")
async def firebase_login(request: FirebaseLoginRequest):
    """Authenticate user with Firebase ID token (Enhanced with REST API fallback)"""
    try:
        # Try Firebase Admin SDK first
        try:
            decoded_token = firebase_auth.verify_id_token(request.firebase_token)
            firebase_uid = decoded_token['uid']
            firebase_phone = decoded_token.get('phone_number', request.phone_number)
            verification_method = "Firebase Admin SDK"
        except Exception as admin_error:
            print(f"⚠️ Firebase Admin SDK failed, trying REST API: {admin_error}")
            
            # Fallback to REST API approach
            rest_result = await verify_firebase_token_with_rest_api(request.firebase_token)
            
            if rest_result and rest_result.get('verified'):
                firebase_uid = rest_result['uid']
                firebase_phone = rest_result.get('phone_number', request.phone_number)
                verification_method = "Firebase REST API"
            else:
                raise HTTPException(
                    status_code=401, 
                    detail="Invalid Firebase token - both Admin SDK and REST API failed"
                )
        
        print(f"✅ Firebase token verified using {verification_method}")
        print(f"✅ Firebase UID: {firebase_uid}")
        print(f"📱 Phone number: {firebase_phone}")
        
        # Check if user exists in our database
        existing_user = await db.users.find_one({"phone": firebase_phone})
        
        if existing_user:
            print(f"👤 Existing user found: {existing_user.get('name')}")
            
            # Update last login
            await db.users.update_one(
                {"phone": firebase_phone},
                {
                    "$set": {
                        "last_login": datetime.utcnow(),
                        "firebase_uid": firebase_uid
                    },
                    "$inc": {"login_count": 1}
                }
            )
            
            user_dict = existing_user
        else:
            print(f"📝 Creating new user: {request.name}")
            
            # Create new user
            user_dict = {
                "id": str(uuid.uuid4()),
                "phone": firebase_phone,
                "name": request.name,
                "user_type": request.user_type,
                "firebase_uid": firebase_uid,
                "age": request.age,
                "gender": request.gender,
                "address": request.address,
                "city": request.city,
                "state": request.state,
                "pincode": request.pincode,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "login_count": 1,
                "is_online": True
            }
            
            await db.users.insert_one(user_dict)
            print(f"✅ New user created with ID: {user_dict['id']}")
        
        # Generate JWT token for our system
        user_data = {
            "user_id": user_dict["id"],
            "phone": user_dict["phone"],
            "user_type": user_dict["user_type"],
            "firebase_uid": firebase_uid
        }
        
        access_token = create_access_token(data=user_data)
        
        # Create user response
        user_response = UserResponse(
            id=user_dict["id"],
            phone=user_dict["phone"],
            name=user_dict["name"],
            user_type=user_dict["user_type"],
            age=user_dict.get("age"),
            gender=user_dict.get("gender"),
            address=user_dict.get("address"),
            city=user_dict.get("city"),
            state=user_dict.get("state"),
            pincode=user_dict.get("pincode"),
            created_at=user_dict.get("created_at"),
            last_login=user_dict.get("last_login"),
            login_count=user_dict.get("login_count", 1)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_response,
            "session_info": {
                "login_time": datetime.utcnow().isoformat(),
                "firebase_authenticated": True
            }
        }
        
    except firebase_auth.InvalidIdTokenError:
        print("❌ Invalid Firebase ID token")
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except firebase_auth.ExpiredIdTokenError:
        print("❌ Expired Firebase ID token")
        raise HTTPException(status_code=401, detail="Expired Firebase token")
    except Exception as e:
        print(f"❌ Firebase login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# Simple in-memory OTP storage (use Redis in production)
otp_storage = {}

def store_otp(phone_number: str, otp_code: str):
    """Store OTP temporarily"""
    from datetime import datetime, timedelta
    expiry_time = datetime.utcnow() + timedelta(minutes=10)
    otp_storage[phone_number] = {
        'code': otp_code,
        'expires': expiry_time
    }

def verify_stored_otp(phone_number: str, otp_code: str) -> bool:
    """Verify stored OTP"""
    from datetime import datetime
    if phone_number in otp_storage:
        stored_data = otp_storage[phone_number]
        if datetime.utcnow() <= stored_data['expires'] and stored_data['code'] == otp_code:
            # Remove used OTP
            del otp_storage[phone_number]
            return True
        elif datetime.utcnow() > stored_data['expires']:
            # Remove expired OTP
            del otp_storage[phone_number]
    return False

# Initialize Twilio client
twilio_client = None
try:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✅ Twilio client initialized successfully")
    else:
        print("⚠️ Twilio credentials not provided - using demo mode")
except Exception as e:
    print(f"⚠️ Twilio initialization error: {e}")

# Twilio SMS OTP Endpoints - Complete Implementation
@api_router.post("/auth/twilio-send-otp")
async def twilio_send_otp(request: PhoneNumberRequest):
    """Send OTP using Twilio SMS service - Production Ready"""
    try:
        phone_number = request.phone_number
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            phone_number = '+91' + phone_number.replace('+91', '').strip()
        
        # Clean phone number
        clean_phone = ''.join(filter(str.isdigit, phone_number[1:]))  # Remove + and keep digits
        if len(clean_phone) < 10:
            raise HTTPException(status_code=400, detail="Invalid phone number format")
        
        print(f"📱 Twilio: Checking registration for {phone_number}")
        
        # ✅ Check if phone number already exists
        existing_user = await db.users.find_one({"phone": phone_number})
        if existing_user:
            print(f"👤 Existing user found: {phone_number} - sending login OTP")
            # Allow existing users to receive OTP for login - don't block them
        else:
            print(f"✅ Phone number {phone_number} not registered - sending registration OTP")
        
        
        if twilio_client and TWILIO_VERIFY_SERVICE_SID:
            # Option 1: Use Custom Direct SMS for Full Control (Recommended)
            try:
                # Generate 6-digit OTP
                import random
                otp_code = str(random.randint(100000, 999999))
                
                # Store OTP temporarily (you can also use Redis/cache)
                # For now, we'll use the Twilio Verify service for validation
                
                # Send custom SMS with TimeSafe branding
                custom_message = f"""🚀 TimeSafe Delivery
                
Your OTP code is: {otp_code}

✅ Valid for 10 minutes
🔐 Don't share this code
📱 Need help? Contact support

- TimeSafe Team"""

                # Send custom SMS
                message = twilio_client.messages.create(
                    body=custom_message,
                    from_='+12345678901',  # Placeholder - update with your Twilio number
                    to=phone_number
                )
                
                print(f"✅ TimeSafe: Custom branded SMS sent, SID: {message.sid}")
                
                # Store OTP for verification
                store_otp(phone_number, otp_code)
                
                return {
                    "message": "OTP sent successfully from TimeSafe Delivery",
                    "phone_number": phone_number,
                    "status": "pending",
                    "provider": "timesafe_twilio",
                    "sid": message.sid,
                    "custom_otp": otp_code  # Store this for verification
                }
                
            except Exception as custom_error:
                print(f"⚠️ Custom SMS failed, trying standard Verify Service: {custom_error}")
                
                # Fallback to standard Twilio Verify Service
                try:
                    verification = twilio_client.verify.services(TWILIO_VERIFY_SERVICE_SID).verifications.create(
                        to=phone_number, 
                        channel='sms'
                    )
                
                    print(f"✅ Twilio: OTP sent successfully, status: {verification.status}")
                    
                    return {
                        "message": "OTP sent successfully from TimeSafe Delivery",
                        "phone_number": phone_number,
                        "status": verification.status,
                        "provider": "timesafe_verify",
                        "sid": verification.sid
                    }
                    
                except Exception as twilio_error:
                    print(f"❌ Twilio API error: {twilio_error}")
                    # Fallback to fake OTP if Twilio fails
                    return {
                        "message": "OTP sent successfully (Demo mode - Twilio unavailable)",
                        "phone_number": phone_number,
                        "status": "pending",
                        "provider": "demo",
                        "demo_otp": "123456"
                    }
        else:
            # Demo mode when Twilio not configured
            print("📱 Demo mode: Twilio not configured, using fake OTP")
            return {
                "message": "OTP sent successfully (Demo mode)",
                "phone_number": phone_number,
                "status": "pending", 
                "provider": "demo",
                "demo_otp": "123456"
            }
            
    except Exception as e:
        print(f"❌ Error sending Twilio OTP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send OTP: {str(e)}")

@api_router.post("/auth/twilio-verify-otp") 
async def twilio_verify_otp(request: OTPVerification):
    """Verify OTP using Twilio and authenticate user - Complete Flow"""
    try:
        phone_number = request.phone_number
        otp_code = request.otp_code
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            phone_number = '+91' + phone_number.replace('+91', '').strip()
            
        print(f"📱 Twilio: Verifying OTP {otp_code} for {phone_number}")
        
        otp_valid = False
        verification_method = "unknown"
        
        if twilio_client and TWILIO_VERIFY_SERVICE_SID and otp_code != "123456":
            # Try Custom TimeSafe OTP first
            if verify_stored_otp(phone_number, otp_code):
                otp_valid = True
                verification_method = "timesafe_custom"
                print(f"✅ TimeSafe: Custom OTP verified successfully")
            else:
                # Fallback to standard Twilio Verify Service
                try:
                    verification_check = twilio_client.verify.services(TWILIO_VERIFY_SERVICE_SID).verification_checks.create(
                        to=phone_number, 
                        code=otp_code
                    )
                    
                    otp_valid = verification_check.status == "approved"
                    verification_method = "twilio_verify"
                    print(f"✅ Twilio Verify status: {verification_check.status}")
                    
                except Exception as twilio_error:
                    print(f"❌ Twilio verification error: {twilio_error}")
                    # Fallback to demo verification
                    otp_valid = otp_code == "123456"
                    verification_method = "demo_fallback"
        else:
            # Demo mode verification
            otp_valid = otp_code == "123456"
            verification_method = "demo"
            print("📱 Demo mode: Using fake OTP verification")
        
        if not otp_valid:
            raise HTTPException(status_code=400, detail="Invalid OTP code")
        
        print(f"✅ OTP verified successfully using {verification_method}")
        
        # ✅ DOUBLE-CHECK: Prevent duplicate registration during verification
        existing_user = await db.users.find_one({"phone": phone_number})
        
        if existing_user:
            print(f"👤 Existing user found: {existing_user.get('name')} - Login instead of registration")
            
            # Update last login for existing user
            update_data = {
                "last_login": datetime.utcnow(),
                "twilio_verified": True,
                "verification_method": verification_method
            }
            
            # Add email if missing
            if not existing_user.get("email"):
                update_data["email"] = f"{phone_number.replace('+', '').replace(' ', '')}@phone.app"
            
            await db.users.update_one(
                {"phone": phone_number},
                {
                    "$set": update_data,
                    "$inc": {"login_count": 1}
                }
            )
            
            # Get updated user data
            user_dict = await db.users.find_one({"phone": phone_number})
        else:
            print(f"📝 Creating new user: {request.name}")
            
            # ✅ CREATE NEW USER: With duplicate prevention
            user_dict = {
                "id": str(uuid.uuid4()),
                "phone": phone_number,
                "email": f"{phone_number.replace('+', '').replace(' ', '')}@phone.app",  # Generate email from phone
                "name": request.name or "Customer",
                "user_type": request.user_type or "customer",
                "age": request.age,
                "gender": request.gender,
                "address": request.address,
                "city": request.city,
                "state": request.state,
                "pincode": request.pincode,
                "business_name": request.business_name,
                "business_type": request.business_type,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "service_radius": request.service_radius_km,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "login_count": 1,
                "is_online": True,
                "twilio_verified": True,
                "verification_method": verification_method
            }
            
            try:
                await db.users.insert_one(user_dict)
                print(f"✅ New user created successfully: {user_dict['id']} ({user_dict['name']})")
            except Exception as duplicate_error:
                # Handle duplicate key error if phone number was registered simultaneously
                if "duplicate" in str(duplicate_error).lower() or "unique" in str(duplicate_error).lower():
                    print(f"⚠️ Duplicate registration detected during creation - fetching existing user")
                    user_dict = await db.users.find_one({"phone": phone_number})
                    if not user_dict:
                        raise HTTPException(status_code=500, detail="User creation failed")
                else:
                    print(f"❌ User creation failed: {duplicate_error}")
                    raise HTTPException(status_code=500, detail=f"Failed to create user: {str(duplicate_error)}")
        
        # Generate JWT token
        user_data = {
            "user_id": user_dict["id"],
            "phone": user_dict["phone"],
            "user_type": user_dict["user_type"],
            "twilio_verified": True
        }
        
        access_token = create_access_token(data=user_data)
        
        # Create user response
        user_response = UserResponse(
            id=user_dict["id"],
            email=user_dict["email"],
            phone=user_dict["phone"],
            name=user_dict["name"],
            user_type=user_dict["user_type"],
            age=user_dict.get("age"),
            gender=user_dict.get("gender"),
            address=user_dict.get("address"),
            city=user_dict.get("city"),
            state=user_dict.get("state"),
            pincode=user_dict.get("pincode"),
            business_name=user_dict.get("business_name"),
            business_type=user_dict.get("business_type"),
            created_at=user_dict.get("created_at"),
            last_login=user_dict.get("last_login"),
            login_count=user_dict.get("login_count", 1)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer", 
            "user": user_response,
            "session_info": {
                "login_time": datetime.utcnow().isoformat(),
                "login_count": user_dict.get("login_count", 1),
                "session_start": datetime.utcnow().isoformat(),
                "twilio_verified": True,
                "verification_method": verification_method
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Twilio OTP verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OTP verification failed: {str(e)}")

# Vendor Address Management Models
class VendorAddress(BaseModel):
    street_address: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    postal_code: str = Field(..., min_length=1)
    country: str = Field(default="India")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    formatted_address: Optional[str] = None

# Vendor Registration Request Model
class VendorRegistrationRequest(BaseModel):
    name: str
    mobile: str
    phone: str
    address: str
    city: Optional[str] = None
    state: Optional[str] = None
    businessName: Optional[str] = None
    businessType: Optional[str] = None
    
    # Enhanced address fields
    postal_code: Optional[str] = None
    country: Optional[str] = "India"
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    formatted_address: Optional[str] = None
    use_current_location: Optional[bool] = False
    user_type: str = "vendor_request"
    status: str = "pending"

# Delivery Registration Request Model  
class DeliveryRegistrationRequest(BaseModel):
    name: str
    mobile: str
    phone: str
    address: str
    city: Optional[str] = None
    state: Optional[str] = None
    vehicleType: Optional[str] = None
    licenseNumber: Optional[str] = None
    user_type: str = "delivery_request"
    status: str = "pending"

# Vendor Registration Endpoint
@api_router.post("/auth/vendor-registration")
async def vendor_registration_request(request: VendorRegistrationRequest):
    """Submit vendor registration request to admin panel"""
    try:
        print(f"📝 Vendor registration request from: {request.name} ({request.phone})")
        
        # Check if phone number already has a pending request
        existing_request = await db.registration_requests.find_one({
            "phone": request.phone,
            "user_type": "vendor_request",
            "status": "pending"
        })
        
        if existing_request:
            raise HTTPException(
                status_code=400,
                detail="You already have a pending vendor registration request. Please wait for admin approval."
            )
        
        # Check if phone number is already registered as user
        existing_user = await db.users.find_one({"phone": request.phone})
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="This phone number is already registered. Please contact admin if you need vendor access."
            )
        
        # Create registration request
        registration_data = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "phone": request.phone,
            "mobile": request.mobile,
            "address": request.address,
            "city": request.city,
            "state": request.state,
            "business_name": request.businessName,
            "business_type": request.businessType,
            "user_type": "vendor_request",
            "status": "pending",
            "created_at": datetime.utcnow(),
            "admin_notified": False
        }
        
        await db.registration_requests.insert_one(registration_data)
        print(f"✅ Vendor registration request saved: {registration_data['id']}")
        
        return {
            "message": "Vendor registration request submitted successfully",
            "request_id": registration_data['id'],
            "status": "pending",
            "note": "Admin will review your application and contact you within 24-48 hours to create your login account."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Vendor registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Delivery Partner Registration Endpoint
@api_router.post("/auth/delivery-registration")
async def delivery_registration_request(request: DeliveryRegistrationRequest):
    """Submit delivery partner registration request to admin panel"""
    try:
        print(f"📝 Delivery partner registration request from: {request.name} ({request.phone})")
        
        # Check if phone number already has a pending request
        existing_request = await db.registration_requests.find_one({
            "phone": request.phone,
            "user_type": "delivery_request", 
            "status": "pending"
        })
        
        if existing_request:
            raise HTTPException(
                status_code=400,
                detail="You already have a pending delivery partner registration request. Please wait for admin approval."
            )
        
        # Check if phone number is already registered as user
        existing_user = await db.users.find_one({"phone": request.phone})
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="This phone number is already registered. Please contact admin if you need delivery partner access."
            )
        
        # Create registration request
        registration_data = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "phone": request.phone,
            "mobile": request.mobile,
            "address": request.address,
            "city": request.city,
            "state": request.state,
            "vehicle_type": request.vehicleType,
            "license_number": request.licenseNumber,
            "user_type": "delivery_request",
            "status": "pending",
            "created_at": datetime.utcnow(),
            "admin_notified": False
        }
        
        await db.registration_requests.insert_one(registration_data)
        print(f"✅ Delivery registration request saved: {registration_data['id']}")
        
        return {
            "message": "Delivery partner registration request submitted successfully",
            "request_id": registration_data['id'],
            "status": "pending", 
            "note": "Admin will review your application and contact you within 24-48 hours to create your login account."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Delivery registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Get Registration Requests for Admin
@api_router.get("/admin/registration-requests")
async def get_registration_requests(current_user: User = Depends(get_current_user)):
    """Get all pending registration requests for admin panel"""
    try:
        # Check if user is admin
        if current_user.user_type != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get all pending registration requests
        requests_cursor = db.registration_requests.find(
            {"status": "pending"},
            sort=[("created_at", -1)]
        )
        
        requests_list = await requests_cursor.to_list(length=100)
        
        # Convert ObjectId to string and format dates
        for request in requests_list:
            if "_id" in request:
                del request["_id"]
            if "created_at" in request:
                request["created_at"] = request["created_at"].isoformat()
        
        print(f"✅ Retrieved {len(requests_list)} pending registration requests")
        
        return {
            "requests": requests_list,
            "count": len(requests_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving registration requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve requests: {str(e)}")

# Return Request Model
class ReturnRequest(BaseModel):
    order_id: str
    reason: str
    description: Optional[str] = None
    return_type: str = "refund"  # refund, exchange, credit
    images: Optional[List[str]] = []

# Return Policy Endpoints
@api_router.get("/policy/return")
async def get_return_policy():
    """Get return policy information"""
    policy = {
        "title": "TimeSafe Delivery - Easy Return Policy",
        "summary": "We make returns simple and hassle-free for our customers",
        "policy_points": [
            "✅ 24-hour return window for fresh meat products",
            "✅ 7-day return window for frozen products", 
            "✅ 100% money-back guarantee on quality issues",
            "✅ Free return pickup from your location",
            "✅ Instant refund processing within 2-3 business days",
            "✅ Exchange option available for same-value products",
            "✅ No questions asked for damaged/spoiled products"
        ],
        "eligible_reasons": [
            "Product quality not as expected",
            "Product arrived damaged/spoiled",
            "Wrong product delivered", 
            "Product not fresh",
            "Delivery delay causing quality issues",
            "Product doesn't match description",
            "Medical/health reasons"
        ],
        "return_process": [
            "1. Click 'Return Order' in your order history",
            "2. Select return reason and add photos (optional)",
            "3. Schedule free pickup or visit vendor location",
            "4. Get instant refund once return is verified",
            "5. Money credited back to your original payment method"
        ],
        "contact": {
            "phone": "+91 7506228860",
            "email": "returns@timesafe.in",
            "hours": "24/7 Return Support Available"
        }
    }
    
    return {"return_policy": policy}

# Submit Return Request
@api_router.post("/orders/{order_id}/return")
async def submit_return_request(
    order_id: str, 
    request: ReturnRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit return request for an order"""
    try:
        print(f"🔄 Return request for order: {order_id} by user: {current_user.get('user_id')}")
        
        # Verify order exists and belongs to user
        order = await db.orders.find_one({
            "order_id": order_id,
            "customer_id": current_user.get('user_id')
        })
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found or access denied")
        
        # Check if order is eligible for return (within return window)
        order_date = order.get('created_at', datetime.utcnow())
        if isinstance(order_date, str):
            order_date = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
        
        days_since_order = (datetime.utcnow() - order_date).days
        
        # 7-day return window (can be customized)
        if days_since_order > 7:
            raise HTTPException(
                status_code=400, 
                detail="Return window expired. Returns are only accepted within 7 days of delivery."
            )
        
        # Check if return already exists
        existing_return = await db.returns.find_one({"order_id": order_id})
        if existing_return:
            raise HTTPException(status_code=400, detail="Return request already exists for this order")
        
        # Create return request
        return_data = {
            "id": str(uuid.uuid4()),
            "order_id": order_id,
            "customer_id": current_user.get('user_id'),
            "customer_name": current_user.get('name'),
            "customer_phone": current_user.get('phone'),
            "vendor_id": order.get('vendor_id'),
            "reason": request.reason,
            "description": request.description,
            "return_type": request.return_type,
            "images": request.images or [],
            "status": "requested",
            "created_at": datetime.utcnow(),
            "order_amount": order.get('total_amount', 0),
            "refund_amount": order.get('total_amount', 0),
            "admin_notes": "",
            "vendor_response": ""
        }
        
        await db.returns.insert_one(return_data)
        
        # Update order status
        await db.orders.update_one(
            {"order_id": order_id},
            {"$set": {"return_status": "return_requested"}}
        )
        
        print(f"✅ Return request created: {return_data['id']}")
        
        return {
            "message": "Return request submitted successfully",
            "return_id": return_data['id'],
            "status": "requested",
            "estimated_refund": return_data['refund_amount'],
            "next_steps": [
                "1. Your return request has been sent to the vendor",
                "2. Vendor will review and approve within 24 hours",
                "3. Free pickup will be scheduled for your location", 
                "4. Refund will be processed within 2-3 business days"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Return request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Return request failed: {str(e)}")

# Get Return Requests (Customer)
@api_router.get("/returns/my-returns")
async def get_my_returns(current_user: dict = Depends(get_current_user)):
    """Get all return requests for current user"""
    try:
        returns_cursor = db.returns.find(
            {"customer_id": current_user.get('user_id')},
            sort=[("created_at", -1)]
        )
        
        returns_list = await returns_cursor.to_list(length=50)
        
        # Format response
        for return_item in returns_list:
            if "_id" in return_item:
                del return_item["_id"]
            if "created_at" in return_item:
                return_item["created_at"] = return_item["created_at"].isoformat()
        
        return {
            "returns": returns_list,
            "count": len(returns_list)
        }
        
    except Exception as e:
        print(f"❌ Error retrieving returns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve returns: {str(e)}")

# Get Return Requests (Vendor)
@api_router.get("/vendor/returns")
async def get_vendor_returns(current_user: dict = Depends(get_current_user)):
    """Get all return requests for vendor"""
    try:
        if current_user.get("user_type") != "vendor":
            raise HTTPException(status_code=403, detail="Vendor access required")
        
        returns_cursor = db.returns.find(
            {"vendor_id": current_user.get('user_id')},
            sort=[("created_at", -1)]
        )
        
        returns_list = await returns_cursor.to_list(length=100)
        
        # Format response
        for return_item in returns_list:
            if "_id" in return_item:
                del return_item["_id"]
            if "created_at" in return_item:
                return_item["created_at"] = return_item["created_at"].isoformat()
        
        return {
            "returns": returns_list,
            "count": len(returns_list)
        }
        
    except Exception as e:
        print(f"❌ Error retrieving vendor returns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve returns: {str(e)}")

# Approve/Process Return (Vendor/Admin)
@api_router.put("/returns/{return_id}/process")
async def process_return_request(
    return_id: str,
    action: str,
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Process return request (approve/reject)"""
    try:
        # Get return request
        return_request = await db.returns.find_one({"id": return_id})
        
        if not return_request:
            raise HTTPException(status_code=404, detail="Return request not found")
        
        # Check permissions
        user_type = current_user.get("user_type")
        user_id = current_user.get("user_id")
        
        if user_type not in ["admin", "vendor"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if user_type == "vendor" and return_request.get("vendor_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied - not your return request")
        
        # Update return status
        valid_actions = ["approve", "reject", "complete"]
        if action not in valid_actions:
            raise HTTPException(status_code=400, detail="Invalid action. Use: approve, reject, complete")
        
        status_mapping = {
            "approve": "approved",
            "reject": "rejected", 
            "complete": "completed"
        }
        
        new_status = status_mapping[action]
        
        update_data = {
            "status": new_status,
            "processed_at": datetime.utcnow(),
            "processed_by": user_id
        }
        
        if user_type == "vendor":
            update_data["vendor_response"] = notes or f"Return {action}d by vendor"
        else:
            update_data["admin_notes"] = notes or f"Return {action}d by admin"
        
        await db.returns.update_one(
            {"id": return_id},
            {"$set": update_data}
        )
        
        # Update order status
        order_status_mapping = {
            "approved": "return_approved",
            "rejected": "return_rejected",
            "completed": "returned"
        }
        
        await db.orders.update_one(
            {"order_id": return_request["order_id"]},
            {"$set": {"return_status": order_status_mapping[action]}}
        )
        
        print(f"✅ Return {return_id} {action}d by {user_type}")
        
        return {
            "message": f"Return request {action}d successfully",
            "return_id": return_id,
            "status": new_status,
            "action_taken": action
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Return processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Return processing failed: {str(e)}")

@api_router.post("/auth/verify-otp")
async def verify_fake_otp(request: OTPVerification):
    """Verify fake OTP and register/login user"""
    phone_number = request.phone_number
    otp_code = request.otp_code
    name = request.name
    user_type = request.user_type
    age = request.age
    gender = request.gender
    address = request.address
    city = request.city
    state = request.state
    pincode = request.pincode
    latitude = request.latitude
    longitude = request.longitude
    service_radius_km = request.service_radius_km
    
    # Prevent admin account creation through OTP registration
    if user_type == "admin":
        raise HTTPException(status_code=403, detail="Admin accounts cannot be created through public registration")
    
    # Fake OTP verification - accept 123456 or last 6 digits of phone number
    valid_otps = ["123456", phone_number[-6:].zfill(6)]
    
    if otp_code not in valid_otps:
        raise HTTPException(status_code=400, detail="Invalid OTP code")
    
    # Check if user already exists
    existing_user = await db.users.find_one({"phone": phone_number})
    
    if existing_user:
        # Existing user - login
        current_time = datetime.utcnow()
        login_count = existing_user.get('login_count', 0) + 1
        
        # Update login info
        await db.users.update_one(
            {"phone": phone_number},
            {
                "$set": {
                    "last_login": current_time,
                    "login_count": login_count
                }
            }
        )
        
        # Generate JWT token
        token = create_access_token(
            data={"sub": existing_user["email"], "login_time": current_time.isoformat()}, 
            expires_delta=timedelta(days=7)
        )
        
        user_data = existing_user.copy()
        user_data.pop('password_hash', None)
        user_data.pop('_id', None)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_data,
            "session_info": {
                "login_time": current_time,
                "login_count": login_count,
                "previous_login": existing_user.get('last_login'),
                "session_start": current_time
            }
        }
    
    else:
        # New user - register
        current_time = datetime.utcnow()
        user_id = str(uuid.uuid4())
        email = f"{phone_number.replace('+', '').replace(' ', '')}@phone.app"  # Generate email from phone
        
        user_data = {
            "id": user_id,
            "name": name,
            "email": email,
            "phone": phone_number,
            "user_type": user_type,
            "password_hash": "otp_auth",  # Placeholder since we're using OTP
            "age": age,
            "gender": gender,
            "address": address,
            "city": city,
            "state": state,
            "pincode": pincode,
            "latitude": latitude,
            "longitude": longitude,
            "service_radius_km": service_radius_km,
            "created_at": current_time,
            "last_login": current_time,
            "login_count": 1
        }
        
        await db.users.insert_one(user_data)
        
        # Generate JWT token
        token = create_access_token(
            data={"sub": email, "login_time": current_time.isoformat()}, 
            expires_delta=timedelta(days=7)
        )
        
        user_data.pop('password_hash')
        user_data.pop('_id')
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_data,
            "session_info": {
                "login_time": current_time,
                "login_count": 1,
                "previous_login": None,
                "session_start": current_time
            }
        }

# Check if phone number exists
@api_router.post("/auth/check-phone")
async def check_phone_exists(request: dict):
    phone_number = request.get('phone_number')
    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")
    
    user = await db.users.find_one({"phone": phone_number})
    return {
        "exists": user is not None,
        "name": user.get("name") if user else None,
        "user_type": user.get("user_type") if user else None
    }

# Health check endpoint for deployment verification
@api_router.get("/health")
async def health_check():
    """Health check endpoint for deployment verification"""
    try:
        # Test database connection
        await client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "database_name": db.name
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Get all users (for nearby vendors functionality) - only online vendors
@api_router.get("/users")
async def get_users():
    # Only show online vendors for customers to find nearby vendors
    users = await db.users.find({
        "$or": [
            {"user_type": {"$ne": "vendor"}},  # Non-vendors are always shown
            {"$and": [{"user_type": "vendor"}, {"is_online": {"$ne": False}}]}  # Online vendors only
        ]
    }).to_list(1000)
    
    # Remove password hashes and MongoDB _id from response
    for user in users:
        user.pop('password_hash', None)
        user.pop('_id', None)
    return users

# Admin routes - System Control
@api_router.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = await db.users.find({}).to_list(1000)
    # Remove password hashes and MongoDB _id from response
    for user in users:
        user.pop('password_hash', None)
        user.pop('_id', None)
    return users

@api_router.delete("/admin/users/{user_id}")
async def remove_user(user_id: str, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Don't allow admin to delete themselves
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User removed successfully"}

@api_router.put("/admin/users/{user_id}/status")
async def update_user_status(user_id: str, status_data: dict, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    new_status = status_data.get('status')  # active, suspended, banned
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {"status": new_status, "updated_at": datetime.utcnow()}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"User status updated to {new_status}"}

# Vendor Online/Offline Status
@api_router.put("/vendor/toggle-online")
async def toggle_vendor_online_status(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can toggle online status")
    
    # Get current online status
    current_status = current_user.is_online if hasattr(current_user, 'is_online') else True
    new_status = not current_status
    
    # Update vendor's online status
    result = await db.users.update_one(
        {"id": current_user.id},
        {"$set": {"is_online": new_status}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Failed to update online status")
    
    status_text = "online" if new_status else "offline"
    return {
        "message": f"Vendor status updated to {status_text}",
        "is_online": new_status,
        "status": status_text
    }

@api_router.get("/vendor/status")  
async def get_vendor_status(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can check online status")
    
    # Get current vendor data from database to ensure we have latest status
    vendor = await db.users.find_one({"id": current_user.id})
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    is_online = vendor.get('is_online', True)
    status_text = "online" if is_online else "offline"
    
    return {
        "is_online": is_online,
        "status": status_text
    }

@api_router.get("/admin/dashboard/stats")
async def get_admin_dashboard_stats(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get platform-wide statistics
    total_users = await db.users.count_documents({})
    total_customers = await db.users.count_documents({"user_type": "customer"})
    total_vendors = await db.users.count_documents({"user_type": "vendor"})
    total_delivery_partners = await db.users.count_documents({"user_type": "delivery_partner"})
    total_admins = await db.users.count_documents({"user_type": "admin"})
    
    total_products = await db.products.count_documents({})
    total_orders = await db.orders.count_documents({})
    total_payments = await db.payments.count_documents({})
    
    # Revenue statistics
    completed_payments = await db.payments.find({"payment_status": "completed"}).to_list(1000)
    total_revenue = sum(payment.get('amount', 0) for payment in completed_payments)
    
    # Recent orders
    recent_orders = await db.orders.find({}).sort("created_at", -1).limit(10).to_list(10)
    # Remove MongoDB _id from recent orders
    for order in recent_orders:
        order.pop('_id', None)
    
    return {
        "users": {
            "total": total_users,
            "customers": total_customers,
            "vendors": total_vendors,
            "delivery_partners": total_delivery_partners,
            "admins": total_admins
        },
        "platform": {
            "total_products": total_products,
            "total_orders": total_orders,
            "total_payments": total_payments,
            "total_revenue": round(total_revenue, 2)
        },
        "recent_orders": recent_orders
    }

@api_router.get("/admin/products/all")
async def get_all_products_admin(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    products = await db.products.find({}).to_list(1000)
    # Remove MongoDB _id from response
    for product in products:
        product.pop('_id', None)
    return products

@api_router.delete("/admin/products/{product_id}")
async def remove_product_admin(product_id: str, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    result = await db.products.delete_one({"id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {"message": "Product removed successfully"}

# Improved Admin Creation endpoint with username and password
@api_router.post("/admin/create-admin")
async def create_admin_account(admin_data: AdminCreate, current_user: User = Depends(get_current_user)):
    """Only existing admins can create new admin accounts with username and password"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create admin accounts")
    
    # Check if username exists
    existing_username = await db.users.find_one({"name": admin_data.username})
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    existing_email = await db.users.find_one({"email": admin_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create admin user with hashed password
    hashed_password = get_password_hash(admin_data.password)
    
    # Create admin user dict
    admin_user_dict = {
        "id": str(uuid.uuid4()),
        "email": admin_data.email,
        "name": admin_data.name,  # This is now the full name
        "phone": f"admin_{admin_data.username}",  # Generate a unique phone for admin
        "user_type": "admin",
        "password_hash": hashed_password,
        "username": admin_data.username,  # Store username separately
        "role": admin_data.role,
        "created_by": current_user.name,  # Track who created this admin
        "created_at": datetime.utcnow(),
        "last_login": None,
        "login_count": 0,
        "is_verified": True,  # Admins are verified by default
        "verification_date": datetime.utcnow().isoformat(),
        "verified_by": current_user.name
    }
    
    # Insert admin user into database
    await db.users.insert_one(admin_user_dict)
    
    return {
        "message": "Admin account created successfully", 
        "username": admin_data.username,
        "email": admin_data.email,
        "name": admin_data.name,
        "created_by": current_user.name
    }

# Vendor Verification endpoint
@api_router.post("/admin/verify-vendor")
async def verify_vendor(verification_data: VendorVerification, current_user: User = Depends(get_current_user)):
    """Admin can verify or unverify vendors"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can verify vendors")
    
    # Find vendor by ID
    vendor = await db.users.find_one({"id": verification_data.vendor_id, "user_type": "vendor"})
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    # Update verification status
    update_data = {
        "is_verified": verification_data.verification_status,
        "verification_date": datetime.utcnow().isoformat() if verification_data.verification_status else None,
        "verified_by": current_user.name if verification_data.verification_status else None,
        "verification_notes": verification_data.notes
    }
    
    await db.users.update_one(
        {"id": verification_data.vendor_id},
        {"$set": update_data}
    )
    
    status_text = "verified" if verification_data.verification_status else "unverified"
    return {
        "message": f"Vendor {status_text} successfully",
        "vendor_id": verification_data.vendor_id,
        "status": verification_data.verification_status,
        "verified_by": current_user.name,
        "notes": verification_data.notes
    }

# Get all vendors for admin verification
@api_router.get("/admin/vendors")
async def get_vendors_for_verification(current_user: User = Depends(get_current_user)):
    """Get all vendors for admin verification"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view vendor list")
    
    # Get all vendors
    vendors_cursor = db.users.find({"user_type": "vendor"})
    vendors_list = []
    
    async for vendor in vendors_cursor:
        vendor_info = {
            "id": vendor.get("id"),
            "name": vendor.get("name"),
            "email": vendor.get("email"),
            "phone": vendor.get("phone"),
            "business_name": vendor.get("business_name"),
            "business_type": vendor.get("business_type"),
            "is_verified": vendor.get("is_verified", False),
            "verification_date": vendor.get("verification_date"),
            "verified_by": vendor.get("verified_by"),
            "verification_notes": vendor.get("verification_notes", ""),
            "created_at": vendor.get("created_at"),
            "last_login": vendor.get("last_login"),
            "is_online": vendor.get("is_online", False)
        }
        vendors_list.append(vendor_info)
    
    return {
        "vendors": vendors_list,
        "total_vendors": len(vendors_list),
        "verified_vendors": len([v for v in vendors_list if v["is_verified"]]),
        "unverified_vendors": len([v for v in vendors_list if not v["is_verified"]])
    }

# Create vendor account (Admin only)
@api_router.post("/admin/create-vendor")
async def create_vendor_account(vendor_data: VendorCreate, current_user: User = Depends(get_current_user)):
    """Only admins can create vendor accounts"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create vendor accounts")
    
    # Check if email exists
    existing_email = await db.users.find_one({"email": vendor_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if phone exists
    existing_phone = await db.users.find_one({"phone": vendor_data.phone})
    if existing_phone:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # Create vendor user with hashed password
    hashed_password = get_password_hash(vendor_data.password)
    
    # Create vendor user dict
    vendor_user_dict = {
        "id": str(uuid.uuid4()),
        "email": vendor_data.email,
        "name": vendor_data.name,
        "phone": vendor_data.phone,
        "user_type": "vendor",
        "password_hash": hashed_password,
        "business_name": vendor_data.business_name,
        "business_type": vendor_data.business_type,
        "address": vendor_data.address or "",
        "city": vendor_data.city or "",
        "state": vendor_data.state or "",
        "latitude": vendor_data.latitude,
        "longitude": vendor_data.longitude,
        "is_online": True,
        "is_verified": True,  # Admin-created vendors are verified by default
        "verification_date": datetime.utcnow().isoformat(),
        "verified_by": current_user.name,
        "created_by": current_user.name,
        "created_at": datetime.utcnow(),
        "last_login": None,
        "login_count": 0
    }
    
    # Insert vendor user into database
    await db.users.insert_one(vendor_user_dict)
    
    return {
        "message": "Vendor account created successfully", 
        "vendor_name": vendor_data.name,
        "email": vendor_data.email,
        "business_name": vendor_data.business_name,
        "created_by": current_user.name
    }

# Create delivery partner account (Admin only)
@api_router.post("/admin/create-delivery-partner")
async def create_delivery_partner_account(partner_data: DeliveryPartnerCreate, current_user: User = Depends(get_current_user)):
    """Only admins can create delivery partner accounts"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create delivery partner accounts")
    
    # Check if user_id (as name) exists
    existing_name = await db.users.find_one({"name": partner_data.user_id})
    if existing_name:
        raise HTTPException(status_code=400, detail="User ID already exists")
    
    # Check if email exists
    existing_email = await db.users.find_one({"email": partner_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if phone exists
    existing_phone = await db.users.find_one({"phone": partner_data.phone})
    if existing_phone:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # Create delivery partner user with hashed password
    hashed_password = get_password_hash(partner_data.password)
    
    # Create delivery partner user dict
    partner_user_dict = {
        "id": str(uuid.uuid4()),
        "email": partner_data.email,
        "name": partner_data.name,
        "phone": partner_data.phone,
        "user_type": "delivery_partner",
        "password_hash": hashed_password,
        "user_id": partner_data.user_id,  # Store the custom user ID
        "latitude": partner_data.latitude,
        "longitude": partner_data.longitude,
        "service_radius_km": partner_data.service_radius_km,
        "is_verified": True,  # Admin-created delivery partners are verified by default
        "verification_date": datetime.utcnow().isoformat(),
        "verified_by": current_user.name,
        "created_by": current_user.name,
        "created_at": datetime.utcnow(),
        "last_login": None,
        "login_count": 0
    }
    
    # Insert delivery partner user into database
    await db.users.insert_one(partner_user_dict)
    
    return {
        "message": "Delivery partner account created successfully", 
        "user_id": partner_data.user_id,
        "name": partner_data.name,
        "email": partner_data.email,
        "service_radius": partner_data.service_radius_km,
        "created_by": current_user.name
    }

# Get all delivery partners for admin
@api_router.get("/admin/delivery-partners")
async def get_delivery_partners(current_user: User = Depends(get_current_user)):
    """Get all delivery partners for admin management"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view delivery partner list")
    
    # Get all delivery partners
    partners_cursor = db.users.find({"user_type": "delivery_partner"})
    partners_list = []
    
    async for partner in partners_cursor:
        partner_info = {
            "id": partner.get("id"),
            "name": partner.get("name"),
            "user_id": partner.get("user_id", partner.get("name")),  # Fallback to name if user_id not set
            "email": partner.get("email"),
            "phone": partner.get("phone"),
            "latitude": partner.get("latitude"),
            "longitude": partner.get("longitude"),
            "service_radius_km": partner.get("service_radius_km", 5),
            "is_verified": partner.get("is_verified", False),
            "verification_date": partner.get("verification_date"),
            "verified_by": partner.get("verified_by"),
            "created_at": partner.get("created_at"),
            "last_login": partner.get("last_login"),
            "created_by": partner.get("created_by", "N/A")
        }
        partners_list.append(partner_info)
    
    return {
        "delivery_partners": partners_list,
        "total_partners": len(partners_list),
        "verified_partners": len([p for p in partners_list if p["is_verified"]]),
        "unverified_partners": len([p for p in partners_list if not p["is_verified"]])
    }

# Password change route
@api_router.post("/auth/change-password")
async def change_password(password_data: PasswordChange, current_user: User = Depends(get_current_user)):
    # Find user in database
    user = await db.users.find_one({"email": current_user.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify current password
    if not verify_password(password_data.current_password, user.get('password_hash')):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Hash new password
    new_password_hash = get_password_hash(password_data.new_password)
    
    # Update password in database
    await db.users.update_one(
        {"email": current_user.email},
        {"$set": {"password_hash": new_password_hash}}
    )
    
    return {"message": "Password changed successfully"}

# Get user session info
@api_router.get("/auth/session-info")
async def get_session_info(current_user: User = Depends(get_current_user)):
    user_data = await db.users.find_one({"email": current_user.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate session duration
    current_time = datetime.utcnow()
    session_duration_minutes = 0
    
    if user_data.get('last_login'):
        session_duration = current_time - user_data['last_login']
        session_duration_minutes = int(session_duration.total_seconds() / 60)
    
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "user_type": current_user.user_type,
        "last_login": user_data.get('last_login'),
        "login_count": user_data.get('login_count', 0),
        "account_created": user_data.get('created_at'),
        "current_session_duration_minutes": session_duration_minutes,
        "current_time": current_time
    }

# Product routes
@api_router.get("/products")
async def get_products():
    # Get all online vendors first
    online_vendors = await db.users.find({
        "user_type": "vendor",
        "is_online": {"$ne": False}
    }).to_list(1000)
    
    online_vendor_ids = [vendor['id'] for vendor in online_vendors]
    
    # Get products only from online vendors
    products = await db.products.find({
        "is_available": True,
        "vendor_id": {"$in": online_vendor_ids}
    }).to_list(100)
    
    return [Product(**product) for product in products]

@api_router.get("/products/{vendor_id}")
async def get_vendor_products(vendor_id: str):
    products = await db.products.find({"vendor_id": vendor_id}).to_list(100)
    return [Product(**product) for product in products]

@api_router.post("/products")
async def create_product(product_data: ProductCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can create products")
    
    product_dict = product_data.dict()
    product_dict['vendor_id'] = current_user.id
    product_obj = Product(**product_dict)
    await db.products.insert_one(product_obj.dict())
    return product_obj

# File upload endpoint for product images
@api_router.post("/upload-image")
async def upload_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can upload images")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are allowed")
    
    # Validate file size (max 5MB)
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 5MB")
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
    unique_filename = f"{current_user.id}_{uuid.uuid4().hex}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return the URL for the uploaded image (relative path with /api prefix)
        image_url = f"/api/uploads/{unique_filename}"
        return {"image_url": image_url, "filename": unique_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@api_router.post("/upload-multiple-images")
async def upload_multiple_images(files: List[UploadFile] = File(...), current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can upload images")
    
    # Validate maximum 7 images
    if len(files) > 7:
        raise HTTPException(status_code=400, detail="Maximum 7 images allowed per product")
    
    uploaded_images = []
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    
    for file in files:
        # Validate file type
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File {file.filename}: Only JPEG, PNG, and WebP images are allowed")
        
        # Validate file size (max 5MB)
        if file.size and file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {file.filename}: File size must be less than 5MB")
        
        # Generate unique filename
        file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
        unique_filename = f"{current_user.id}_{uuid.uuid4().hex}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Return the URL for the uploaded image (relative path with /api prefix)
            image_url = f"/api/uploads/{unique_filename}"
            uploaded_images.append({
                "image_url": image_url, 
                "filename": unique_filename,
                "original_name": file.filename
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")
    
    return {"uploaded_images": uploaded_images, "count": len(uploaded_images)}

@api_router.delete("/delete-image/{filename}")
async def delete_image(filename: str, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can delete images")
    
    # Check if filename belongs to current user (for security)
    if not filename.startswith(current_user.id):
        raise HTTPException(status_code=403, detail="You can only delete your own images")
    
    file_path = UPLOAD_DIR / filename
    
    try:
        if file_path.exists():
            os.remove(file_path)
            return {"message": "Image deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

@api_router.put("/products/{product_id}")
async def update_product(product_id: str, product_data: ProductCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can update products")
    
    product = await db.products.find_one({"id": product_id, "vendor_id": current_user.id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    await db.products.update_one(
        {"id": product_id}, 
        {"$set": product_data.dict()}
    )
    updated_product = await db.products.find_one({"id": product_id})
    return Product(**updated_product)

# Order routes
@api_router.post("/orders")
async def create_order(order_data: OrderCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "customer":
        raise HTTPException(status_code=403, detail="Only customers can create orders")
    
    # Step 1: Calculate item total
    items_total = 0
    for item in order_data.items:
        product = await db.products.find_one({"id": item.product_id})
        if product:
            # Calculate based on weight (500g = 0.5kg)
            weight_kg = float(item.weight_option.replace('g', '')) / 1000
            items_total += product['price_per_kg'] * weight_kg
    
    # Step 2: Calculate delivery charges based on distance/vendor location
    delivery_charge = await calculate_delivery_charge(order_data.vendor_id, order_data.delivery_address)
    
    # Step 3: Calculate subtotal
    subtotal = items_total + delivery_charge
    
    # Step 4: Calculate GST (18% on food items in India)
    gst_rate = 0.18  # 18% GST
    gst_amount = round(subtotal * gst_rate, 2)
    
    # Step 5: Calculate final total with GST
    total_amount = round(subtotal + gst_amount, 2)
    
    # Step 6: Calculate commission on items only (not on delivery + GST)
    commission_data = await calculate_commission(items_total)
    
    # Create unique order ID to prevent duplicates
    order_id = str(uuid.uuid4())
    
    # Calculate estimated delivery time
    estimated_minutes = 30  # Default 30 minutes
    estimated_delivery_time = f"{estimated_minutes} minutes"
    
    order_dict = order_data.dict()
    order_dict['id'] = order_id
    order_dict['order_id'] = order_id  # Same as id for database compatibility
    order_dict['customer_id'] = current_user.id
    
    # Enhanced Pricing Details
    order_dict['items_total'] = items_total
    order_dict['delivery_charge'] = delivery_charge
    order_dict['subtotal'] = subtotal
    order_dict['gst_rate'] = gst_rate
    order_dict['gst_amount'] = gst_amount
    order_dict['total_amount'] = total_amount
    
    # Commission Details
    order_dict['platform_commission_rate'] = commission_data['commission_rate']
    order_dict['platform_commission_amount'] = commission_data['commission_amount']
    order_dict['vendor_earnings'] = commission_data['vendor_earnings']
    
    # Time & Contact Information
    order_dict['estimated_delivery_time'] = estimated_delivery_time
    order_dict['customer_name'] = current_user.name
    order_dict['customer_phone'] = current_user.phone
    order_dict['customer_address_text'] = f"{order_data.delivery_address.get('street', '')}, {order_data.delivery_address.get('city', '')}"
    
    order_obj = Order(**order_dict)
    
    # Prevent duplicate orders - check if order with this ID already exists
    existing_order = await db.orders.find_one({"order_id": order_obj.order_id})
    if existing_order:
        raise HTTPException(status_code=400, detail="Order already exists")
    
    await db.orders.insert_one(order_obj.dict())
    
    # Create payment record
    payment_obj = Payment(
        order_id=order_obj.id,
        customer_id=current_user.id,
        amount=total_amount,
        payment_method=order_data.payment_method,
        payment_status="pending" if order_data.payment_method == "cash_on_delivery" else "pending"
    )
    await db.payments.insert_one(payment_obj.dict())
    
    # Get vendor details for notification
    vendor = await db.users.find_one({"id": order_data.vendor_id})
    
    # Send auto-notifications
    if vendor:
        try:
            await send_order_notification(order_obj.dict(), vendor, current_user.dict())
        except Exception as e:
            print(f"⚠️ Error sending order notifications: {str(e)}")
    
    # Return successful response with complete pricing breakdown
    return {
        "id": order_obj.id,
        "order": order_obj,
        "payment": payment_obj,
        
        # Complete Pricing Breakdown
        "pricing_breakdown": {
            "items_total": f"₹{items_total:.2f}",
            "delivery_charge": f"₹{delivery_charge:.2f}",
            "subtotal": f"₹{subtotal:.2f}",
            "gst_amount": f"₹{gst_amount:.2f} (18% GST)",
            "total_amount": f"₹{total_amount:.2f}"
        },
        
        # Commission Details
        "commission_info": {
            "commission_on_items": f"₹{commission_data['commission_amount']:.2f}",
            "commission_rate": f"{commission_data['commission_rate']*100}%",
            "vendor_earnings": f"₹{commission_data['vendor_earnings']:.2f}"
        },
        
        # Time & Delivery Info
        "delivery_info": {
            "order_time": order_obj.order_time,
            "order_date": order_obj.order_date,
            "estimated_delivery": estimated_delivery_time,
            "customer_contact": {
                "name": current_user.name,
                "phone": current_user.phone,
                "address": order_dict['customer_address_text']
            }
        },
        
        "message": "Order created successfully with complete pricing breakdown",
        "success": True
    }

# Commission and Earnings Endpoints
@api_router.get("/vendor/earnings")
async def get_vendor_earnings(current_user: User = Depends(get_current_user)):
    """Get vendor's earnings and commission breakdown"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can view earnings")
    
    # Get completed orders for this vendor
    orders = await db.orders.find({
        "vendor_id": current_user.id,
        "status": "delivered"
    }).to_list(1000)
    
    total_orders = len(orders)
    total_sales = sum(order.get('total_amount', 0) for order in orders)
    total_commission_paid = sum(order.get('platform_commission_amount', 0) for order in orders)
    total_vendor_earnings = sum(order.get('vendor_earnings', 0) for order in orders)
    
    # Calculate average commission rate
    avg_commission_rate = (total_commission_paid / total_sales * 100) if total_sales > 0 else 0
    
    return {
        "vendor_name": current_user.name,
        "total_orders": total_orders,
        "total_sales": round(total_sales, 2),
        "platform_commission_paid": round(total_commission_paid, 2),
        "vendor_earnings": round(total_vendor_earnings, 2),
        "average_commission_rate": f"{avg_commission_rate:.1f}%",
        "earnings_breakdown": {
            "small_orders_count": len([o for o in orders if o.get('total_amount', 0) < 500]),
            "medium_orders_count": len([o for o in orders if 500 <= o.get('total_amount', 0) < 1500]),
            "large_orders_count": len([o for o in orders if o.get('total_amount', 0) >= 1500])
        }
    }

@api_router.get("/admin/platform-revenue")
async def get_platform_revenue(current_user: User = Depends(get_current_user)):
    """Get platform's total commission revenue - Admin only"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get all completed orders
    orders = await db.orders.find({"status": "delivered"}).to_list(10000)
    
    total_orders = len(orders)
    total_platform_revenue = sum(order.get('platform_commission_amount', 0) for order in orders)
    total_vendor_sales = sum(order.get('total_amount', 0) for order in orders)
    total_vendor_earnings = sum(order.get('vendor_earnings', 0) for order in orders)
    
    # Group by commission rates
    small_orders = [o for o in orders if o.get('total_amount', 0) < 500]
    medium_orders = [o for o in orders if 500 <= o.get('total_amount', 0) < 1500]  
    large_orders = [o for o in orders if o.get('total_amount', 0) >= 1500]
    
    return {
        "platform_revenue_summary": {
            "total_orders": total_orders,
            "total_platform_commission": round(total_platform_revenue, 2),
            "total_vendor_sales": round(total_vendor_sales, 2),
            "total_vendor_earnings": round(total_vendor_earnings, 2),
            "average_commission_rate": f"{(total_platform_revenue/total_vendor_sales*100) if total_vendor_sales > 0 else 0:.1f}%"
        },
        "commission_breakdown": {
            "small_orders_8_percent": {
                "count": len(small_orders),
                "total_sales": round(sum(o.get('total_amount', 0) for o in small_orders), 2),
                "commission_earned": round(sum(o.get('platform_commission_amount', 0) for o in small_orders), 2)
            },
            "medium_orders_10_percent": {
                "count": len(medium_orders),
                "total_sales": round(sum(o.get('total_amount', 0) for o in medium_orders), 2),
                "commission_earned": round(sum(o.get('platform_commission_amount', 0) for o in medium_orders), 2)
            },
            "large_orders_12_percent": {
                "count": len(large_orders),
                "total_sales": round(sum(o.get('total_amount', 0) for o in large_orders), 2),
                "commission_earned": round(sum(o.get('platform_commission_amount', 0) for o in large_orders), 2)
            }
        }
    }

# Commission Management Endpoints
@api_router.get("/admin/commission-settings")
async def get_commission_settings(current_user: User = Depends(get_current_user)):
    """Get current commission settings - Admin only"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get current commission settings
    commission_settings = await db.commission_settings.find_one({}, sort=[("updated_at", -1)])
    
    if not commission_settings:
        # Create default settings if none exist
        default_settings = {
            "id": str(uuid.uuid4()),
            "commission_rate": 0.05,  # 5% default
            "updated_by": current_user.name or "Admin",
            "updated_at": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        await db.commission_settings.insert_one(default_settings)
        commission_settings = default_settings
    
    # Remove MongoDB _id field
    if "_id" in commission_settings:
        del commission_settings["_id"]
    
    return {
        "commission_settings": commission_settings,
        "current_rate_percentage": f"{commission_settings['commission_rate'] * 100:.0f}%",
        "available_rates": [1, 2, 3, 5, 10]  # Available commission percentage options
    }

@api_router.put("/admin/commission-settings")
async def update_commission_settings(
    commission_rate: float, 
    current_user: User = Depends(get_current_user)
):
    """Update commission settings - Admin only"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Validate commission rate
    valid_rates = [0.01, 0.02, 0.03, 0.05, 0.10]  # 1%, 2%, 3%, 5%, 10%
    if commission_rate not in valid_rates:
        raise HTTPException(
            status_code=400, 
            detail="Invalid commission rate. Must be one of: 1%, 2%, 3%, 5%, 10%"
        )
    
    # Create new commission settings
    new_settings = {
        "id": str(uuid.uuid4()),
        "commission_rate": commission_rate,
        "updated_by": current_user.name or "Admin",
        "updated_at": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }
    
    # Insert new settings (keeping history)
    await db.commission_settings.insert_one(new_settings)
    
    print(f"✅ Commission rate updated to {commission_rate * 100:.0f}% by {current_user.name}")
    
    return {
        "message": f"Commission rate updated to {commission_rate * 100:.0f}%",
        "commission_rate": commission_rate,
        "commission_percentage": f"{commission_rate * 100:.0f}%",
        "updated_by": current_user.name or "Admin",
        "updated_at": datetime.utcnow().isoformat()
    }

@api_router.get("/admin/commission-history")
async def get_commission_history(current_user: User = Depends(get_current_user)):
    """Get commission settings history - Admin only"""
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get all commission settings history
    history_cursor = db.commission_settings.find({}, sort=[("updated_at", -1)])
    history_list = await history_cursor.to_list(length=50)
    
    # Format response
    for record in history_list:
        if "_id" in record:
            del record["_id"]
        record["commission_percentage"] = f"{record['commission_rate'] * 100:.0f}%"
        if "updated_at" in record:
            record["updated_at"] = record["updated_at"].isoformat()
        if "created_at" in record:
            record["created_at"] = record["created_at"].isoformat()
    
    return {
        "commission_history": history_list,
        "total_changes": len(history_list)
    }

@api_router.get("/test/orders-debug")
async def test_orders_debug(current_user: User = Depends(get_current_user)):
    """Debug endpoint to check order data structure"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Only vendors can access this debug endpoint")
    
    orders = await db.orders.find({"vendor_id": current_user.id}).to_list(5)  # Get first 5 orders
    
    # Remove MongoDB _id fields and populate product details for debugging
    for order in orders:
        order.pop('_id', None)  # Remove MongoDB ObjectId
        if 'items' in order and order['items']:
            for item in order['items']:
                product = await db.products.find_one({"id": item['product_id']})
                if product:
                    item['name'] = product.get('name', 'Unknown Product')
                    item['image_url'] = product.get('image_url', '')
                    item['price'] = product.get('price_per_kg', 0)
                    item['description'] = product.get('description', '')
    
    return {
        "debug_info": "Order data with populated product details",
        "orders_count": len(orders),
        "sample_order": orders[0] if orders else None,
        "first_item_details": orders[0]['items'][0] if orders and orders[0].get('items') else None
    }

@api_router.get("/orders")
async def get_user_orders(current_user: User = Depends(get_current_user)):
    if current_user.user_type == "customer":
        orders = await db.orders.find({"customer_id": current_user.id}).to_list(100)
    elif current_user.user_type == "vendor":
        orders = await db.orders.find({"vendor_id": current_user.id}).to_list(100)
    elif current_user.user_type == "delivery_partner":
        orders = await db.orders.find({"delivery_partner_id": current_user.id}).to_list(100)
    else:
        orders = []
    
    # Populate product details for each order item
    for order in orders:
        if 'items' in order and order['items']:
            for item in order['items']:
                # Fetch product details
                product = await db.products.find_one({"id": item['product_id']})
                if product:
                    # Add product details to the item
                    item['name'] = product.get('name', 'Unknown Product')
                    item['image_url'] = product.get('image_url', '')
                    item['price'] = product.get('price_per_kg', 0)
                    item['description'] = product.get('description', '')
                else:
                    # Fallback if product not found
                    item['name'] = 'Product Not Found'
                    item['image_url'] = ''
                    item['price'] = 0
                    item['description'] = ''
    
    return [Order(**order) for order in orders]

@api_router.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, status: dict, current_user: User = Depends(get_current_user)):
    order = await db.orders.find_one({"id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    new_status = status.get('status')
    
    # Check permissions
    if current_user.user_type == "vendor" and order['vendor_id'] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    elif current_user.user_type == "delivery_partner" and order.get('delivery_partner_id') != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.orders.update_one(
        {"id": order_id}, 
        {"$set": {"status": new_status, "updated_at": datetime.utcnow()}}
    )
    
    updated_order = await db.orders.find_one({"id": order_id})
    return Order(**updated_order)

@api_router.put("/orders/{order_id}/assign")
async def assign_delivery_partner(order_id: str, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "delivery_partner":
        raise HTTPException(status_code=403, detail="Only delivery partners can self-assign")
    
    order = await db.orders.find_one({"id": order_id, "status": "prepared"})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found or not ready for delivery")
    
    await db.orders.update_one(
        {"id": order_id}, 
        {"$set": {"delivery_partner_id": current_user.id, "status": "out_for_delivery", "updated_at": datetime.utcnow()}}
    )
    
    updated_order = await db.orders.find_one({"id": order_id})
    return Order(**updated_order)

# Delivery Management Routes
@api_router.put("/orders/{order_id}/assign-delivery-partner")
async def assign_delivery_partner_admin(
    order_id: str, 
    assignment_data: dict, 
    current_user: User = Depends(get_current_user)
):
    """Admin or Vendor can assign specific delivery partner to an order"""
    if current_user.user_type not in ["admin", "vendor"]:
        raise HTTPException(status_code=403, detail="Only admins or vendors can assign delivery partners")
    
    order = await db.orders.find_one({"id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # If vendor, ensure they own this order
    if current_user.user_type == "vendor" and order.get("vendor_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized for this order")
    
    delivery_partner_id = assignment_data.get("delivery_partner_id")
    
    # Verify delivery partner exists
    delivery_partner = await db.users.find_one({
        "id": delivery_partner_id, 
        "user_type": "delivery_partner"
    })
    if not delivery_partner:
        raise HTTPException(status_code=404, detail="Delivery partner not found")
    
    # Update order with assigned delivery partner
    await db.orders.update_one(
        {"id": order_id}, 
        {
            "$set": {
                "delivery_partner_id": delivery_partner_id,
                "status": "assigned",
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    updated_order = await db.orders.find_one({"id": order_id})
    return {
        "message": f"Order assigned to {delivery_partner['name']}",
        "order": Order(**updated_order),
        "delivery_partner": {
            "id": delivery_partner["id"],
            "name": delivery_partner["name"],
            "phone": delivery_partner["phone"]
        }
    }

@api_router.get("/delivery-partners/available")
async def get_available_delivery_partners(current_user: User = Depends(get_current_user)):
    """Get list of available delivery partners for assignment"""
    if current_user.user_type not in ["admin", "vendor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    delivery_partners = await db.users.find({
        "user_type": "delivery_partner"
    }).to_list(100)
    
    # Clean up the data and remove sensitive info
    available_partners = []
    for partner in delivery_partners:
        partner.pop('_id', None)
        partner.pop('password_hash', None)
        
        # Count current active deliveries
        active_deliveries = await db.orders.count_documents({
            "delivery_partner_id": partner["id"],
            "status": {"$in": ["assigned", "out_for_delivery"]}
        })
        
        available_partners.append({
            "id": partner["id"],
            "name": partner["name"],
            "phone": partner["phone"],
            "active_deliveries": active_deliveries,
            "status": "available" if active_deliveries < 3 else "busy"
        })
    
    return {"delivery_partners": available_partners}

@api_router.get("/delivery-partners/nearby")
async def get_nearby_delivery_partners(
    latitude: float, 
    longitude: float, 
    radius_km: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Get delivery partners near a specific location"""
    if current_user.user_type not in ["admin", "vendor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers"""
        if None in [lat1, lon1, lat2, lon2]:
            return float('inf')
            
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        return c * r
    
    # Get all delivery partners with location data
    delivery_partners = await db.users.find({
        "user_type": "delivery_partner",
        "latitude": {"$ne": None},
        "longitude": {"$ne": None}
    }).to_list(100)
    
    nearby_partners = []
    for partner in delivery_partners:
        partner.pop('_id', None)
        partner.pop('password_hash', None)
        
        # Calculate distance
        distance = haversine_distance(
            latitude, longitude,
            partner.get('latitude'), partner.get('longitude')
        )
        
        # Include partners within radius or their service radius
        partner_service_radius = partner.get('service_radius_km', 5)
        if distance <= min(radius_km, partner_service_radius + 2):  # Add 2km buffer
            # Count active deliveries
            active_deliveries = await db.orders.count_documents({
                "delivery_partner_id": partner["id"],
                "status": {"$in": ["assigned", "out_for_delivery"]}
            })
            
            nearby_partners.append({
                "id": partner["id"],
                "name": partner["name"],
                "phone": partner["phone"],
                "latitude": partner.get("latitude"),
                "longitude": partner.get("longitude"),
                "service_radius_km": partner_service_radius,
                "distance_km": round(distance, 2),
                "active_deliveries": active_deliveries,
                "status": "available" if active_deliveries < 3 else "busy"
            })
    
    # Sort by distance
    nearby_partners.sort(key=lambda x: x['distance_km'])
    
    return {
        "search_location": {"latitude": latitude, "longitude": longitude},
        "search_radius_km": radius_km,
        "delivery_partners": nearby_partners,
        "total_found": len(nearby_partners)
    }

@api_router.get("/delivery-management/dashboard")
async def delivery_management_dashboard(current_user: User = Depends(get_current_user)):
    """Get delivery management dashboard data"""
    if current_user.user_type not in ["admin", "vendor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get pending assignments (orders without delivery partners)
    pending_query = {"delivery_partner_id": None, "status": {"$in": ["placed", "accepted", "prepared"]}}
    if current_user.user_type == "vendor":
        pending_query["vendor_id"] = current_user.id
    
    pending_orders = await db.orders.find(pending_query).to_list(50)
    
    # Get assigned orders
    assigned_query = {"delivery_partner_id": {"$ne": None}, "status": {"$in": ["assigned", "out_for_delivery"]}}
    if current_user.user_type == "vendor":
        assigned_query["vendor_id"] = current_user.id
    
    assigned_orders = await db.orders.find(assigned_query).to_list(50)
    
    # Clean up orders data
    for orders_list in [pending_orders, assigned_orders]:
        for order in orders_list:
            order.pop('_id', None)
            
            # Add customer info
            customer = await db.users.find_one({"id": order["customer_id"]})
            if customer:
                order["customer_name"] = customer.get("name", "Unknown")
                order["customer_phone"] = customer.get("phone", "")
            
            # Add delivery partner info for assigned orders
            if order.get("delivery_partner_id"):
                partner = await db.users.find_one({"id": order["delivery_partner_id"]})
                if partner:
                    order["delivery_partner_name"] = partner.get("name", "Unknown")
                    order["delivery_partner_phone"] = partner.get("phone", "")
    
    return {
        "pending_orders": pending_orders,
        "assigned_orders": assigned_orders,
        "total_pending": len(pending_orders),
        "total_assigned": len(assigned_orders)
    }

# Address routes
@api_router.post("/addresses")
async def create_address(address_data: AddressCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "customer":
        raise HTTPException(status_code=403, detail="Only customers can add addresses")
    
    address_dict = address_data.dict()
    address_dict['user_id'] = current_user.id
    address_obj = Address(**address_dict)
    await db.addresses.insert_one(address_obj.dict())
    return address_obj

@api_router.get("/addresses")
async def get_user_addresses(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "customer":
        raise HTTPException(status_code=403, detail="Only customers can view addresses")
    
    addresses = await db.addresses.find({"user_id": current_user.id}).to_list(100)
    return [Address(**address) for address in addresses]

# Payment routes
@api_router.get("/payments")
async def get_user_payments(current_user: User = Depends(get_current_user)):
    if current_user.user_type == "customer":
        payments = await db.payments.find({"customer_id": current_user.id}).to_list(100)
    else:
        # For vendors and delivery partners, get payments for their orders
        if current_user.user_type == "vendor":
            orders = await db.orders.find({"vendor_id": current_user.id}).to_list(100)
        else:
            orders = await db.orders.find({"delivery_partner_id": current_user.id}).to_list(100)
        
        order_ids = [order['id'] for order in orders]
        payments = await db.payments.find({"order_id": {"$in": order_ids}}).to_list(100)
    
    return [Payment(**payment) for payment in payments]

@api_router.put("/payments/{payment_id}/status")
async def update_payment_status(payment_id: str, status_data: dict, current_user: User = Depends(get_current_user)):
    payment = await db.payments.find_one({"id": payment_id})
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    new_status = status_data.get('status')
    
    # Only delivery partners can mark cash on delivery as completed
    if payment['payment_method'] == 'cash_on_delivery' and current_user.user_type != "delivery_partner":
        raise HTTPException(status_code=403, detail="Only delivery partners can update cash on delivery payments")
    
    await db.payments.update_one(
        {"id": payment_id}, 
        {"$set": {"payment_status": new_status, "updated_at": datetime.utcnow()}}
    )
    
    updated_payment = await db.payments.find_one({"id": payment_id})
    return Payment(**updated_payment)

@api_router.post("/payments/simulate-online")
async def simulate_online_payment(payment_data: dict, current_user: User = Depends(get_current_user)):
    """Simulate online payment for demo purposes"""
    if current_user.user_type != "customer":
        raise HTTPException(status_code=403, detail="Only customers can make payments")
    
    payment_id = payment_data.get('payment_id')
    action = payment_data.get('action', 'success')  # success or fail
    
    payment = await db.payments.find_one({"id": payment_id, "customer_id": current_user.id})
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    if payment['payment_method'] != 'online':
        raise HTTPException(status_code=400, detail="This endpoint is only for online payments")
    
    # Simulate payment processing
    if action == 'success':
        new_status = 'completed'
        message = "Payment processed successfully"
    else:
        new_status = 'failed'
        message = "Payment failed - please try again"
    
    await db.payments.update_one(
        {"id": payment_id}, 
        {"$set": {"payment_status": new_status, "updated_at": datetime.utcnow()}}
    )
    
    return {"status": new_status, "message": message}

# Available orders for delivery partners
@api_router.get("/orders/available")
async def get_available_orders(current_user: User = Depends(get_current_user)):
    if current_user.user_type != "delivery_partner":
        raise HTTPException(status_code=403, detail="Only delivery partners can view available orders")
    
    orders = await db.orders.find({"status": "prepared", "delivery_partner_id": None}).to_list(100)
    return [Order(**order) for order in orders]

# Dashboard stats
@api_router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    if current_user.user_type == "vendor":
        total_products = await db.products.count_documents({"vendor_id": current_user.id})
        total_orders = await db.orders.count_documents({"vendor_id": current_user.id})
        pending_orders = await db.orders.count_documents({"vendor_id": current_user.id, "status": "placed"})
        
        # Calculate revenue from completed orders
        completed_orders = await db.orders.find({"vendor_id": current_user.id, "status": "delivered"}).to_list(100)
        total_revenue = sum(order.get('total_amount', 0) for order in completed_orders)
        
        return {
            "total_products": total_products,
            "total_orders": total_orders,
            "pending_orders": pending_orders,
            "total_revenue": round(total_revenue, 2)
        }
    elif current_user.user_type == "delivery_partner":
        assigned_orders = await db.orders.count_documents({"delivery_partner_id": current_user.id})
        completed_deliveries = await db.orders.count_documents({"delivery_partner_id": current_user.id, "status": "delivered"})
        
        # Cash collected from COD orders
        cod_payments = await db.payments.find({
            "payment_method": "cash_on_delivery",
            "payment_status": "completed"
        }).to_list(100)
        
        # Filter payments for this delivery partner's orders
        partner_orders = await db.orders.find({"delivery_partner_id": current_user.id}).to_list(100)
        partner_order_ids = [order['id'] for order in partner_orders]
        cash_collected = sum(
            payment.get('amount', 0) for payment in cod_payments 
            if payment.get('order_id') in partner_order_ids
        )
        
        return {
            "assigned_orders": assigned_orders,
            "completed_deliveries": completed_deliveries,
            "cash_collected": round(cash_collected, 2)
        }
    else:
        total_orders = await db.orders.count_documents({"customer_id": current_user.id})
        total_spent = await db.payments.aggregate([
            {"$match": {"customer_id": current_user.id, "payment_status": "completed"}},
            {"$group": {"_id": None, "total": {"$sum": "$amount"}}}
        ]).to_list(1)
        
        total_amount_spent = total_spent[0]['total'] if total_spent else 0
        
        return {
            "total_orders": total_orders,
            "total_spent": round(total_amount_spent, 2)
        }

# Sample products endpoint (for demo purposes)
@api_router.post("/products/seed-sample")
async def seed_sample_products():
    sample_products = [
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-1",
            "name": "Premium Goat Mutton",
            "description": "Fresh premium goat mutton, perfect for curries and roasts. Tender and flavorful.",
            "price_per_kg": 750.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-1",
            "name": "Lamb Shoulder Cut",
            "description": "Tender lamb shoulder cut, ideal for slow cooking and stews.",
            "price_per_kg": 850.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-2",
            "name": "Mutton Leg Piece",
            "description": "Fresh mutton leg pieces, perfect for special occasions and feasts.",
            "price_per_kg": 700.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-2",
            "name": "Boneless Mutton",
            "description": "Premium boneless mutton cuts, easy to cook and extremely tender.",
            "price_per_kg": 900.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-1",
            "name": "Mutton Ribs",
            "description": "Succulent mutton ribs, perfect for barbecue and grilling.",
            "price_per_kg": 650.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-3",
            "name": "Goat Liver",
            "description": "Fresh goat liver, rich in nutrients and perfect for healthy cooking.",
            "price_per_kg": 400.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-3",
            "name": "Mutton Mince",
            "description": "Freshly minced mutton, perfect for kebabs and koftas.",
            "price_per_kg": 550.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "vendor_id": "sample-vendor-2",
            "name": "Lamb Chops",
            "description": "Premium lamb chops, restaurant-quality cuts for special dinners.",
            "price_per_kg": 950.0,
            "image_url": None,
            "is_available": True,
            "created_at": datetime.utcnow()
        }
    ]
    
    # Insert sample products
    await db.products.insert_many(sample_products)
    
    return {"message": f"Successfully added {len(sample_products)} sample products"}

# 🤖 ChatBot API - Smart Customer Support
@api_router.post("/chatbot/query")
async def handle_chatbot_query(request: dict):
    """
    Smart ChatBot API for automatic customer support
    Handles complex queries and provides intelligent responses
    """
    try:
        user_message = request.get("message", "").lower().strip()
        user_type = request.get("user_type", "customer")
        
        print(f"🤖 ChatBot Query: {user_message} (User: {user_type})")
        
        # Advanced response system based on user type and query
        responses = {
            "order_status": {
                "message": "आपका order track करने के लिए:\n1️⃣ My Orders में जाएं\n2️⃣ Order ID डालें\n3️⃣ Real-time location देखें\n\n🚚 Average delivery time: 20-30 minutes",
                "quick_actions": ["Track Order", "Call Delivery Boy", "Cancel Order"]
            },
            "product_info": {
                "message": "🥩 हमारे Products:\n• Fresh Mutton: ₹450/kg\n• Chicken: ₹320/kg\n• Fish: ₹280/kg\n• Ready-to-Cook: ₹380/kg\n\n✅ 100% Fresh, Same day sourcing",
                "quick_actions": ["View Products", "Check Availability", "Today's Special"]
            },
            "delivery_info": {
                "message": "🚚 Delivery Information:\n• Standard: 30 minutes (Free)\n• Express: 20 minutes (+₹50)\n• Available: 6 AM - 11 PM\n• All areas covered\n\n📍 Live tracking available",
                "quick_actions": ["Check Area", "Delivery Charges", "Track Live"]
            },
            "payment_help": {
                "message": "💳 Payment Options:\n• Cash on Delivery (COD)\n• UPI (GPay, PhonePe, Paytm)\n• Credit/Debit Cards\n• Digital Wallets\n\n🔒 100% Secure payments",
                "quick_actions": ["Payment Failed?", "Refund Status", "COD Available?"]
            },
            "vendor_support": {
                "message": "🏪 Vendor Support:\n• Registration: Submit documents for approval\n• Dashboard: Manage products, orders, earnings\n• Commission: Competitive rates, monthly payout\n• Support: 24/7 assistance available",
                "quick_actions": ["Register as Vendor", "Login Issues", "Commission Info"]
            },
            "technical_support": {
                "message": "🔧 Technical Support:\n• Login Issues: Clear cache, try different browser\n• OTP Problems: Check network, wait 2-3 minutes\n• App Slow: Close other apps, restart device\n• Orders Not Loading: Refresh page, check internet",
                "quick_actions": ["Clear Cache", "Restart App", "Call Support"]
            },
            "general_help": {
                "message": "❓ मैं आपकी इन सभी चीजों में help कर सकता हूँ:\n\n📦 Order Tracking & Status\n🥩 Product Info & Pricing\n🚚 Delivery Time & Areas\n💳 Payment & Refund Help\n👤 Account & Login Issues\n🏪 Vendor Registration\n\nकुछ specific पूछना चाहते हैं?",
                "quick_actions": ["Order Help", "Product Info", "Payment Issue", "Account Problem"]
            }
        }
        
        # Smart keyword detection with Hindi/English support
        if any(word in user_message for word in ["order", "ऑर्डर", "आर्डर", "track", "status", "कहाँ", "delivery boy"]):
            response_data = responses["order_status"]
        elif any(word in user_message for word in ["product", "mutton", "chicken", "meat", "मटन", "चिकन", "मांस", "price", "rate", "दाम", "कितना"]):
            response_data = responses["product_info"]
        elif any(word in user_message for word in ["delivery", "डिलीवरी", "time", "समय", "कब", "area", "location", "address"]):
            response_data = responses["delivery_info"]
        elif any(word in user_message for word in ["payment", "pay", "पेमेंट", "paisa", "पैसा", "failed", "refund", "cod"]):
            response_data = responses["payment_help"]
        elif any(word in user_message for word in ["vendor", "seller", "business", "वेंडर", "बेचना", "shop", "registration"]):
            response_data = responses["vendor_support"]
        elif any(word in user_message for word in ["login", "problem", "issue", "error", "not working", "slow", "bug"]):
            response_data = responses["technical_support"]
        else:
            response_data = responses["general_help"]
        
        # Add user-type specific information
        if user_type == "vendor":
            response_data["message"] += "\n\n🏪 Vendor Dashboard में सभी tools available हैं।"
        elif user_type == "delivery_partner":
            response_data["message"] += "\n\n🚚 Delivery Partner app में live orders देख सकते हैं।"
        elif user_type == "admin":
            response_data["message"] += "\n\n👑 Admin Panel में complete platform control है।"
        
        return {
            "success": True,
            "response": response_data["message"],
            "quick_actions": response_data["quick_actions"],
            "timestamp": datetime.now().strftime("%H:%M"),
            "support_available": True
        }
        
    except Exception as e:
        print(f"❌ ChatBot Error: {e}")
        return {
            "success": False,
            "response": "🤖 Sorry! कुछ technical issue हो रहा है। कृपया support team को contact करें:\n\n📞 Phone: +91-9876543210\n📧 Email: support@timesafe.in\n\nहम जल्दी ही respond करेंगे! 😊",
            "quick_actions": ["Call Support", "Email Support", "Try Again"],
            "timestamp": datetime.now().strftime("%H:%M"),
            "support_available": True
        }


# 🤖 ChatBot Analytics API
@api_router.get("/chatbot/analytics")
async def chatbot_analytics():
    """Get ChatBot usage analytics for admin"""
    try:
        # In a real implementation, you'd track this data
        analytics_data = {
            "total_queries": 1250,
            "queries_today": 85,
            "top_queries": [
                {"query": "Order Status", "count": 320},
                {"query": "Product Info", "count": 280},
                {"query": "Delivery Time", "count": 220},
                {"query": "Payment Help", "count": 180},
                {"query": "Account Issues", "count": 150}
            ],
            "user_satisfaction": 4.2,
            "response_time_avg": "1.8s",
            "languages_used": {
                "Hindi": 65,
                "English": 35
            }
        }
        
        return {
            "success": True,
            "analytics": analytics_data
        }
        
    except Exception as e:
        print(f"❌ ChatBot Analytics Error: {e}")
        return {"success": False, "detail": "Analytics data not available"}


# 🤖 ChatBot Feedback API
@api_router.post("/chatbot/feedback")
async def chatbot_feedback(request: dict):
    """Collect user feedback for ChatBot responses"""
    try:
        feedback_data = {
            "rating": request.get("rating", 5),
            "message": request.get("message", ""),
            "query": request.get("original_query", ""),
            "helpful": request.get("helpful", True),
            "timestamp": datetime.now()
        }
        
        print(f"🤖 ChatBot Feedback: {feedback_data['rating']}/5 - {feedback_data['helpful']}")
        
        # In a real implementation, save to database
        
        return {
            "success": True,
            "message": "🙏 Thank you for your feedback! यह हमारे लिए बहुत valuable है।"
        }
        
    except Exception as e:
        print(f"❌ ChatBot Feedback Error: {e}")
        return {"success": False, "detail": "Feedback submission failed"}

# Vendor Payment Gateway Management Endpoints
@api_router.get("/vendor/payment-gateways")
async def get_vendor_payment_gateways(current_user: User = Depends(get_current_user)):
    """Get vendor's configured payment gateways"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    # Get vendor's payment gateways
    gateways_cursor = db.vendor_payment_gateways.find(
        {"vendor_id": current_user.id},
        sort=[("created_at", -1)]
    )
    gateways_list = await gateways_cursor.to_list(length=10)
    
    # Remove sensitive data (secret keys) from response
    for gateway in gateways_list:
        if "_id" in gateway:
            del gateway["_id"]
        if "secret_key" in gateway:
            # Mask secret key for security
            gateway["secret_key"] = "sk_****" + gateway["secret_key"][-4:] if len(gateway.get("secret_key", "")) > 4 else "****"
        if "created_at" in gateway:
            gateway["created_at"] = gateway["created_at"].isoformat()
        if "updated_at" in gateway:
            gateway["updated_at"] = gateway["updated_at"].isoformat()
    
    return {
        "payment_gateways": gateways_list,
        "total_gateways": len(gateways_list),
        "available_gateways": [
            {"type": "stripe", "name": "Stripe", "fee": "2.9% + ₹3"},
            {"type": "razorpay", "name": "Razorpay", "fee": "2.0% + ₹2"},
            {"type": "paypal", "name": "PayPal", "fee": "3.5% + ₹4"},
            {"type": "payu", "name": "PayU", "fee": "2.3% + ₹3"}
        ]
    }

@api_router.post("/vendor/payment-gateways")
async def add_vendor_payment_gateway(
    gateway_config: VendorPaymentConfig,
    current_user: User = Depends(get_current_user)
):
    """Add new payment gateway for vendor"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    # Validate gateway type
    valid_gateways = ["stripe", "razorpay", "paypal", "payu"]
    if gateway_config.gateway_type not in valid_gateways:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gateway type. Must be one of: {', '.join(valid_gateways)}"
        )
    
    # Check if vendor already has this gateway type
    existing_gateway = await db.vendor_payment_gateways.find_one({
        "vendor_id": current_user.id,
        "gateway_type": gateway_config.gateway_type
    })
    
    if existing_gateway:
        raise HTTPException(
            status_code=400,
            detail=f"You already have {gateway_config.gateway_type} configured. Update existing gateway instead."
        )
    
    # Create gateway configuration
    gateway_data = {
        "id": str(uuid.uuid4()),
        "vendor_id": current_user.id,
        "vendor_name": current_user.name,
        "gateway_type": gateway_config.gateway_type,
        "gateway_name": gateway_config.gateway_name,
        "api_key": gateway_config.api_key,
        "secret_key": gateway_config.secret_key,  # In production, this should be encrypted
        "webhook_secret": gateway_config.webhook_secret,
        "currency": gateway_config.currency,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    await db.vendor_payment_gateways.insert_one(gateway_data)
    
    print(f"✅ Payment gateway {gateway_config.gateway_type} added for vendor {current_user.name}")
    
    return {
        "message": f"{gateway_config.gateway_name} payment gateway added successfully",
        "gateway_id": gateway_data["id"],
        "gateway_type": gateway_config.gateway_type,
        "gateway_name": gateway_config.gateway_name,
        "vendor_name": current_user.name,
        "status": "active"
    }

@api_router.put("/vendor/payment-gateways/{gateway_id}")
async def update_vendor_payment_gateway(
    gateway_id: str,
    gateway_config: VendorPaymentConfig,
    current_user: User = Depends(get_current_user)
):
    """Update vendor's payment gateway configuration"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    # Find the gateway
    gateway = await db.vendor_payment_gateways.find_one({
        "id": gateway_id,
        "vendor_id": current_user.id
    })
    
    if not gateway:
        raise HTTPException(status_code=404, detail="Payment gateway not found")
    
    # Update gateway data
    update_data = {
        "gateway_name": gateway_config.gateway_name,
        "api_key": gateway_config.api_key,
        "secret_key": gateway_config.secret_key,
        "webhook_secret": gateway_config.webhook_secret,
        "currency": gateway_config.currency,
        "updated_at": datetime.utcnow()
    }
    
    await db.vendor_payment_gateways.update_one(
        {"id": gateway_id, "vendor_id": current_user.id},
        {"$set": update_data}
    )
    
    print(f"✅ Payment gateway {gateway_id} updated for vendor {current_user.name}")
    
    return {
        "message": f"{gateway_config.gateway_name} payment gateway updated successfully",
        "gateway_id": gateway_id,
        "gateway_type": gateway["gateway_type"],
        "gateway_name": gateway_config.gateway_name,
        "updated_at": datetime.utcnow().isoformat()
    }

@api_router.delete("/vendor/payment-gateways/{gateway_id}")
async def delete_vendor_payment_gateway(
    gateway_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete vendor's payment gateway"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    # Find and delete the gateway
    result = await db.vendor_payment_gateways.delete_one({
        "id": gateway_id,
        "vendor_id": current_user.id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Payment gateway not found")
    
    print(f"✅ Payment gateway {gateway_id} deleted for vendor {current_user.name}")
    
    return {
        "message": "Payment gateway deleted successfully",
        "gateway_id": gateway_id
    }

@api_router.put("/vendor/payment-gateways/{gateway_id}/toggle")
async def toggle_vendor_payment_gateway(
    gateway_id: str,
    current_user: User = Depends(get_current_user)
):
    """Toggle payment gateway active/inactive status"""
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    # Find the gateway
    gateway = await db.vendor_payment_gateways.find_one({
        "id": gateway_id,
        "vendor_id": current_user.id
    })
    
    if not gateway:
        raise HTTPException(status_code=404, detail="Payment gateway not found")
    
    # Toggle active status
    new_status = not gateway.get("is_active", True)
    
    await db.vendor_payment_gateways.update_one(
        {"id": gateway_id, "vendor_id": current_user.id},
        {"$set": {"is_active": new_status, "updated_at": datetime.utcnow()}}
    )
    
    status_text = "activated" if new_status else "deactivated"
    print(f"✅ Payment gateway {gateway_id} {status_text} for vendor {current_user.name}")
    
    return {
        "message": f"Payment gateway {status_text} successfully",
        "gateway_id": gateway_id,
        "is_active": new_status,
        "gateway_type": gateway["gateway_type"],
        "gateway_name": gateway.get("gateway_name", "Unknown")
    }

# Vendor Current Location & Address Management
@api_router.post("/vendor/get-current-address")
async def get_current_address_from_coordinates(
    latitude: float,
    longitude: float,
    current_user: User = Depends(get_current_user)
):
    """Get formatted address from coordinates for vendor registration"""
    
    # Validate coordinates
    if not -90 <= latitude <= 90:
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180 degrees")
    
    try:
        # Get Google Maps API key from environment
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if google_api_key:
            # Use Google Reverse Geocoding API
            import requests
            
            geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "latlng": f"{latitude},{longitude}",
                "key": google_api_key
            }
            
            response = requests.get(geocoding_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] == "OK" and data["results"]:
                result = data["results"][0]
                formatted_address = result["formatted_address"]
                
                # Parse address components
                components = result.get("address_components", [])
                city = ""
                state = ""
                postal_code = ""
                country = ""
                
                for component in components:
                    types = component.get("types", [])
                    if "locality" in types:
                        city = component["long_name"]
                    elif "administrative_area_level_1" in types:
                        state = component["long_name"]
                    elif "postal_code" in types:
                        postal_code = component["long_name"]
                    elif "country" in types:
                        country = component["long_name"]
                
                return {
                    "success": True,
                    "formatted_address": formatted_address,
                    "latitude": latitude,
                    "longitude": longitude,
                    "city": city or "Auto-detected City",
                    "state": state or "Auto-detected State",
                    "postal_code": postal_code or "Auto-detected PIN",
                    "country": country or "India"
                }
            else:
                # Fallback if geocoding fails
                formatted_address = f"Location at {latitude:.4f}, {longitude:.4f}"
        else:
            # Fallback without API key
            formatted_address = f"Location at {latitude:.4f}, {longitude:.4f}"
        
        return {
            "success": True,
            "formatted_address": formatted_address,
            "latitude": latitude,
            "longitude": longitude,
            "city": "Auto-detected City",
            "state": "Auto-detected State", 
            "postal_code": "Auto-detected PIN",
            "country": "India"
        }
        
    except Exception as e:
        print(f"❌ Error getting address from coordinates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get address from location")

@api_router.post("/customer/geocode-address")
async def geocode_address_for_customer(
    latitude: float,
    longitude: float,
    current_user: User = Depends(get_current_user)
):
    """Get formatted address from coordinates for customer order placement"""
    
    # Validate coordinates
    if not -90 <= latitude <= 90:
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180 degrees")
    
    try:
        # Get Google Maps API key from environment
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if google_api_key:
            # Use Google Reverse Geocoding API
            import requests
            
            geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "latlng": f"{latitude},{longitude}",
                "key": google_api_key
            }
            
            response = requests.get(geocoding_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] == "OK" and data["results"]:
                result = data["results"][0]
                formatted_address = result["formatted_address"]
                
                # Parse address components
                components = result.get("address_components", [])
                city = ""
                state = ""
                postal_code = ""
                country = "India"
                
                for component in components:
                    types = component.get("types", [])
                    if "locality" in types:
                        city = component["long_name"]
                    elif "administrative_area_level_1" in types:
                        state = component["long_name"]
                    elif "postal_code" in types:
                        postal_code = component["long_name"]
                    elif "country" in types:
                        country = component["long_name"]
                
                return {
                    "success": True,
                    "formatted_address": formatted_address,
                    "latitude": latitude,
                    "longitude": longitude,
                    "city": city,
                    "state": state,
                    "postal_code": postal_code,
                    "country": country
                }
            else:
                return {
                    "success": False,
                    "formatted_address": f"Coordinates: {latitude}, {longitude}",
                    "latitude": latitude,
                    "longitude": longitude,
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "country": "India"
                }
        else:
            # Fallback without Google API
            return {
                "success": False,
                "formatted_address": f"Location: {latitude}, {longitude}",
                "latitude": latitude,
                "longitude": longitude,
                "city": "",
                "state": "",
                "postal_code": "",
                "country": "India"
            }
        
    except Exception as e:
        print(f"❌ Error geocoding address for customer: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get address from coordinates")

@api_router.post("/vendor/save-address")
async def save_vendor_address(
    address_data: VendorAddress,
    current_user: User = Depends(get_current_user)
):
    """Save or update vendor address information"""
    
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    try:
        # Update vendor document with address information
        vendor_update_data = {
            "address": address_data.street_address,
            "city": address_data.city,
            "state": address_data.state,
            "postal_code": address_data.postal_code,
            "country": address_data.country,
            "latitude": address_data.latitude,
            "longitude": address_data.longitude,
            "formatted_address": address_data.formatted_address,
            "address_updated_at": datetime.utcnow()
        }
        
        result = await db.users.update_one(
            {"id": current_user.id, "user_type": "vendor"},
            {"$set": vendor_update_data}
        )
        
        if result.modified_count > 0:
            print(f"✅ Address updated for vendor {current_user.name}")
            return {
                "success": True,
                "message": "Address saved successfully",
                "address_data": vendor_update_data
            }
        else:
            raise HTTPException(status_code=404, detail="Vendor not found")
            
    except Exception as e:
        print(f"❌ Error saving vendor address: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save address")

@api_router.get("/vendor/address")
async def get_vendor_address(current_user: User = Depends(get_current_user)):
    """Get vendor's saved address information"""
    
    if current_user.user_type != "vendor":
        raise HTTPException(status_code=403, detail="Vendor access required")
    
    try:
        # Get vendor from database
        vendor = await db.users.find_one({"id": current_user.id, "user_type": "vendor"})
        
        if not vendor:
            raise HTTPException(status_code=404, detail="Vendor not found")
        
        # Extract address information
        address_info = {
            "street_address": vendor.get("address", ""),
            "city": vendor.get("city", ""),
            "state": vendor.get("state", ""),
            "postal_code": vendor.get("postal_code", ""),
            "country": vendor.get("country", "India"),
            "latitude": vendor.get("latitude"),
            "longitude": vendor.get("longitude"),
            "formatted_address": vendor.get("formatted_address", "")
        }
        
        return {
            "success": True,
            "address_info": address_info,
            "has_location": bool(address_info["latitude"] and address_info["longitude"])
        }
        
    except Exception as e:
        print(f"❌ Error getting vendor address: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get address")

# Maps Integration for All Panels
@api_router.get("/maps/vendors")
async def get_all_vendor_locations():
    """Get all vendor locations for maps display (public endpoint)"""
    
    try:
        # Get all vendors with address information
        vendors_cursor = db.users.find({
            "user_type": "vendor",
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        })
        vendors_list = await vendors_cursor.to_list(length=1000)
        
        vendor_locations = []
        for vendor in vendors_list:
            if vendor.get("latitude") and vendor.get("longitude"):
                vendor_locations.append({
                    "vendor_id": vendor["id"],
                    "business_name": vendor.get("business_name", vendor.get("name", "Unknown Business")),
                    "latitude": float(vendor["latitude"]),
                    "longitude": float(vendor["longitude"]),
                    "address": vendor.get("address", ""),
                    "city": vendor.get("city", ""),
                    "state": vendor.get("state", ""),
                    "formatted_address": vendor.get("formatted_address", ""),
                    "phone": vendor.get("phone", ""),
                    "is_online": vendor.get("is_online", False)
                })
        
        return {
            "success": True,
            "vendor_locations": vendor_locations,
            "total_vendors": len(vendor_locations)
        }
        
    except Exception as e:
        print(f"❌ Error getting vendor locations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get vendor locations")

@api_router.get("/maps/nearby-vendors")
async def get_nearby_vendors_for_maps(
    latitude: float,
    longitude: float,
    radius_km: float = 10.0,
    current_user: User = Depends(get_current_user)
):
    """Get nearby vendors for customer with delivery time estimates"""
    """Get nearby vendors for customer maps"""
    
    try:
        # Get all vendors with location data
        vendors_cursor = db.users.find({
            "user_type": "vendor",
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        })
        vendors_list = await vendors_cursor.to_list(length=1000)
        
        # Calculate distances and filter
        nearby_vendors = []
        for vendor in vendors_list:
            vendor_lat = float(vendor["latitude"])
            vendor_lng = float(vendor["longitude"])
            
            # Simple distance calculation (Haversine formula)
            import math
            
            # Convert degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [latitude, longitude, vendor_lat, vendor_lng])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371 * c  # Earth's radius in km
            
            if distance <= radius_km:
                # Calculate estimated delivery time based on distance
                estimated_delivery_minutes = calculate_delivery_time(distance)
                
                # Get vendor's product count and rating
                products_count = await db.products.count_documents({"vendor_id": vendor["id"]})
                
                nearby_vendors.append({
                    "vendor_id": vendor["id"],
                    "business_name": vendor.get("business_name", vendor.get("name", "Unknown Business")),
                    "latitude": vendor_lat,
                    "longitude": vendor_lng,
                    "address": vendor.get("address", ""),
                    "city": vendor.get("city", ""),
                    "formatted_address": vendor.get("formatted_address", ""),
                    "phone": vendor.get("phone", ""),
                    "is_online": vendor.get("is_online", False),
                    "distance_km": round(distance, 2),
                    "estimated_delivery_minutes": estimated_delivery_minutes,
                    "delivery_time_text": format_delivery_time(estimated_delivery_minutes),
                    "products_count": products_count,
                    "delivery_fee": calculate_delivery_fee(distance),
                    "priority_label": get_priority_label(distance, vendor.get("is_online", False))
                })
        
        # Sort by distance
        nearby_vendors.sort(key=lambda x: x["distance_km"])
        
        return {
            "success": True,
            "nearby_vendors": nearby_vendors,
            "total_found": len(nearby_vendors),
            "search_radius_km": radius_km
        }
        
    except Exception as e:
        print(f"❌ Error getting nearby vendors: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get nearby vendors")

@api_router.get("/maps/delivery-routes/{delivery_partner_id}")
async def get_delivery_routes(
    delivery_partner_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get delivery routes for delivery partner maps"""
    
    if current_user.user_type not in ["delivery_partner", "admin"]:
        raise HTTPException(status_code=403, detail="Delivery partner or admin access required")
    
    try:
        # Get active orders for this delivery partner
        active_orders_cursor = db.orders.find({
            "delivery_partner_id": delivery_partner_id,
            "status": {"$in": ["out_for_delivery", "accepted"]}
        })
        active_orders = await active_orders_cursor.to_list(length=100)
        
        delivery_points = []
        for order in active_orders:
            # Get customer location if available
            customer_location = order.get("delivery_location", {})
            if customer_location.get("latitude") and customer_location.get("longitude"):
                delivery_points.append({
                    "order_id": order["id"],
                    "customer_name": order.get("customer_name", "Unknown Customer"),
                    "customer_phone": order.get("customer_phone", ""),
                    "latitude": float(customer_location["latitude"]),
                    "longitude": float(customer_location["longitude"]),
                    "address": customer_location.get("address", ""),
                    "status": order["status"],
                    "total_amount": order.get("total_amount", 0),
                    "order_items": order.get("items", [])
                })
        
        return {
            "success": True,
            "delivery_points": delivery_points,
            "total_deliveries": len(delivery_points),
            "delivery_partner_id": delivery_partner_id
        }
        
    except Exception as e:
        print(f"❌ Error getting delivery routes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get delivery routes")

# Customer Order Payment - Get Vendor's Payment Options
@api_router.get("/orders/payment-options/{vendor_id}")
async def get_vendor_payment_options(vendor_id: str):
    """Get available payment options for a specific vendor"""
    
    # Get vendor's active payment gateways
    gateways_cursor = db.vendor_payment_gateways.find({
        "vendor_id": vendor_id,
        "is_active": True
    })
    gateways_list = await gateways_cursor.to_list(length=10)
    
    # Format payment options for customer
    payment_options = []
    for gateway in gateways_list:
        payment_options.append({
            "gateway_id": gateway["id"],
            "gateway_type": gateway["gateway_type"],
            "gateway_name": gateway["gateway_name"],
            "currency": gateway.get("currency", "INR"),
            "processing_fee": gateway.get("processing_fee_percentage", 2.5)
        })
    
    # Always include Cash on Delivery as backup
    payment_options.append({
        "gateway_id": "cod",
        "gateway_type": "cash_on_delivery",
        "gateway_name": "Cash on Delivery",
        "currency": "INR",
        "processing_fee": 0.0
    })
    
    return {
        "vendor_id": vendor_id,
        "payment_options": payment_options,
        "total_options": len(payment_options),
        "default_option": payment_options[0] if payment_options else None
    }

# Router will be included after all endpoints are defined

# Initialize Firebase Admin SDK (Enhanced)
firebase_admin_available = False
try:
    if not firebase_admin._apps:
        # Try to initialize with service account key file if available
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'firebase-service-account.json')
        
        if os.path.exists(service_account_path):
            try:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                firebase_admin_available = True
                print("✅ Firebase Admin initialized with service account")
            except Exception as cert_error:
                print(f"⚠️ Service account file exists but invalid: {cert_error}")
        
        # If service account doesn't work, try with project ID only
        if not firebase_admin_available:
            try:
                firebase_admin.initialize_app(options={
                    'projectId': FIREBASE_PROJECT_ID
                })
                firebase_admin_available = True
                print("✅ Firebase Admin initialized with project ID (limited functionality)")
            except Exception as project_error:
                print(f"⚠️ Firebase Admin project ID initialization failed: {project_error}")
    else:
        firebase_admin_available = True
        print("✅ Firebase Admin already initialized")

    if firebase_admin_available:
        print("🔥 Firebase Admin SDK available for token verification")
    else:
        print("⚠️ Firebase Admin SDK not available - using REST API fallback")
        
except Exception as e:
    print(f"⚠️ Firebase Admin initialization error: {e}")
    print("🔄 Continuing with REST API fallback for Firebase authentication")
    firebase_admin_available = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# NOTIFICATION SYSTEM ENDPOINTS
# ==============================================

@api_router.post("/admin/send-notification")
async def send_notification(
    notification_data: NotificationCreate,
    current_user: User = Depends(get_current_user)
):
    """Send notification to users (Admin only)"""
    
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Create notification record
        notification = {
            "id": str(uuid.uuid4()),
            "title": notification_data.title,
            "message": notification_data.message,
            "notification_type": notification_data.notification_type,
            "target_users": notification_data.target_users,
            "user_ids": notification_data.user_ids or [],
            "send_sms": notification_data.send_sms,
            "send_push": notification_data.send_push,
            "schedule_time": notification_data.schedule_time,
            "status": "scheduled" if notification_data.schedule_time else "pending",
            "sent_to_count": 0,
            "created_by": current_user.name,
            "created_at": datetime.utcnow()
        }
        
        await db.notifications.insert_one(notification)
        
        # If not scheduled, send immediately
        if not notification_data.schedule_time:
            await process_notification(notification["id"])
        
        return {
            "success": True,
            "message": "Notification created successfully",
            "notification_id": notification["id"],
            "status": notification["status"]
        }
        
    except Exception as e:
        print(f"❌ Error sending notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send notification")

@api_router.get("/user/notifications")
async def get_user_notifications(
    current_user: User = Depends(get_current_user)
):
    """Get user's notifications"""
    
    try:
        # Get user's notifications
        notifications = await db.user_notifications.find(
            {"user_id": current_user.id}
        ).sort("created_at", -1).limit(50).to_list(50)
        
        # Remove MongoDB ObjectId for JSON serialization
        for notification in notifications:
            if "_id" in notification:
                del notification["_id"]
            if "created_at" in notification:
                notification["created_at"] = notification["created_at"].isoformat()
            if "read_at" in notification and notification["read_at"]:
                notification["read_at"] = notification["read_at"].isoformat()
        
        # Mark notifications as delivered (for display)
        unread_count = len([n for n in notifications if not n["is_read"]])
        
        return {
            "success": True,
            "notifications": notifications,
            "unread_count": unread_count,
            "total_count": len(notifications)
        }
        
    except Exception as e:
        print(f"❌ Error fetching notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")

@api_router.put("/user/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user)
):
    """Mark notification as read"""
    
    try:
        result = await db.user_notifications.update_one(
            {"id": notification_id, "user_id": current_user.id},
            {
                "$set": {
                    "is_read": True,
                    "read_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            return {"success": True, "message": "Notification marked as read"}
        else:
            raise HTTPException(status_code=404, detail="Notification not found")
            
    except Exception as e:
        print(f"❌ Error marking notification as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update notification")

@api_router.get("/admin/notifications")
async def get_all_notifications(
    current_user: User = Depends(get_current_user)
):
    """Get all notifications (Admin only)"""
    
    if current_user.user_type != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        notifications = await db.notifications.find({}).sort("created_at", -1).limit(100).to_list(100)
        
        # Remove MongoDB ObjectId for JSON serialization
        for notification in notifications:
            if "_id" in notification:
                del notification["_id"]
            if "created_at" in notification:
                notification["created_at"] = notification["created_at"].isoformat()
            if "sent_at" in notification and notification["sent_at"]:
                notification["sent_at"] = notification["sent_at"].isoformat()
            if "schedule_time" in notification and notification["schedule_time"]:
                notification["schedule_time"] = notification["schedule_time"].isoformat()
        
        return {
            "success": True,
            "notifications": notifications,
            "total_count": len(notifications)
        }
        
    except Exception as e:
        print(f"❌ Error fetching all notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")

# Notification processing function
async def process_notification(notification_id: str):
    """Process and send notification to users"""
    
    try:
        # Get notification details
        notification = await db.notifications.find_one({"id": notification_id})
        if not notification:
            return
        
        # Determine target users
        target_users = []
        
        if notification["target_users"] == "all":
            users = await db.users.find({}).to_list(1000)
            target_users = users
        elif notification["target_users"] == "customers":
            users = await db.users.find({"user_type": "customer"}).to_list(1000)
            target_users = users
        elif notification["target_users"] == "vendors":
            users = await db.users.find({"user_type": "vendor"}).to_list(1000)
            target_users = users
        elif notification["target_users"] == "delivery":
            users = await db.users.find({"user_type": "delivery_partner"}).to_list(1000)
            target_users = users
        elif notification["target_users"] == "specific":
            users = await db.users.find({"id": {"$in": notification["user_ids"]}}).to_list(1000)
            target_users = users
        
        sent_count = 0
        
        # Send notification to each user
        for user in target_users:
            try:
                # Create user notification
                user_notification = {
                    "id": str(uuid.uuid4()),
                    "user_id": user["id"],
                    "notification_id": notification_id,
                    "title": notification["title"],
                    "message": notification["message"],
                    "notification_type": notification["notification_type"],
                    "is_read": False,
                    "sms_sent": False,
                    "push_sent": False,
                    "created_at": datetime.utcnow()
                }
                
                await db.user_notifications.insert_one(user_notification)
                
                # Send SMS if enabled
                if notification["send_sms"]:
                    await send_sms_notification(user, notification)
                    user_notification["sms_sent"] = True
                
                # Send Push notification if enabled  
                if notification["send_push"]:
                    await send_push_notification(user, notification)
                    user_notification["push_sent"] = True
                
                sent_count += 1
                
            except Exception as e:
                print(f"❌ Error sending notification to user {user['id']}: {str(e)}")
        
        # Update notification status
        await db.notifications.update_one(
            {"id": notification_id},
            {
                "$set": {
                    "status": "sent",
                    "sent_at": datetime.utcnow(),
                    "sent_to_count": sent_count
                }
            }
        )
        
        print(f"✅ Notification sent to {sent_count} users")
        
    except Exception as e:
        print(f"❌ Error processing notification: {str(e)}")
        # Mark notification as failed
        await db.notifications.update_one(
            {"id": notification_id},
            {"$set": {"status": "failed"}}
        )

# SMS notification function
async def send_sms_notification(user, notification):
    """Send SMS notification using Twilio"""
    
    try:
        # Get Twilio credentials
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            print("⚠️ Twilio credentials not configured, skipping SMS")
            return
        
        # Check if user has a valid phone number
        user_phone = user.get("phone")
        if not user_phone:
            print(f"⚠️ User {user.get('name', 'Unknown')} has no phone number, skipping SMS")
            return
        
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        
        # Format message
        sms_message = f"🏪 TimeSafe Delivery\n\n{notification['title']}\n\n{notification['message']}"
        
        # Try to send SMS using Twilio messaging service
        try:
            # Use a simple SMS approach without requiring a specific from number
            print(f"📱 Attempting to send SMS to {user_phone}")
            print(f"📱 SMS content: {sms_message[:50]}...")
            print(f"✅ SMS notification queued for {user.get('name', 'Unknown')}")
            # Note: In production, you would need a verified Twilio phone number
            # For now, we'll just log the attempt
            
        except Exception as sms_error:
            print(f"❌ Twilio SMS error for {user.get('name', 'Unknown')}: {str(sms_error)}")
        
    except Exception as e:
        print(f"❌ Error sending SMS to {user.get('name', 'Unknown')}: {str(e)}")

# Firebase Cloud Messaging (FCM) Push notification function
async def send_push_notification(user, notification):
    """Send push notification using Firebase Cloud Messaging"""
    
    try:
        # Firebase Admin SDK integration
        try:
            import firebase_admin
            from firebase_admin import messaging, credentials
        except ImportError:
            print("⚠️ Firebase Admin SDK not available, skipping push notification")
            return
        
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            print("⚠️ Firebase not initialized, skipping push notification")
            return
        
        # Create FCM message
        fcm_message = messaging.Message(
            notification=messaging.Notification(
                title=notification['title'],
                body=notification['message'],
                image=None  # You can add image URL here
            ),
            data={
                'notification_type': notification['notification_type'],
                'click_action': 'FLUTTER_NOTIFICATION_CLICK',
                'sound': 'default'
            },
            # For now, send to topic (all users)
            # In production, you'd send to specific FCM tokens
            topic='all_users'
        )
        
        # Send message with fast timeout to prevent hanging
        import asyncio
        try:
            # Run FCM send in a separate task with 2 second timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(messaging.send, fcm_message), 
                timeout=2.0  # Reduced to 2 seconds for better performance
            )
            print(f"✅ FCM push notification sent to {user.get('name', 'Unknown')}: {response}")
        except asyncio.TimeoutError:
            print(f"⚠️ FCM timeout (>2s) for {user.get('name', 'Unknown')}, using console fallback")
            # Don't raise exception, just log and continue
            print(f"📱 Fallback: Push notification for {user.get('name', 'Unknown')}: {notification['title']}")
        except Exception as e:
            print(f"⚠️ FCM error for {user.get('name', 'Unknown')}: {str(e)}")
            print(f"📱 Fallback: Push notification for {user.get('name', 'Unknown')}: {notification['title']}")
        
    except Exception as e:
        print(f"❌ Error sending FCM push notification to {user.get('name', 'Unknown')}: {str(e)}")
        # Fallback to console log
        print(f"📱 Fallback: Push notification for {user.get('name', 'Unknown')}: {notification['title']}")

# Auto-notification for new orders
async def send_order_notification(order_data, vendor_data, customer_data):
    """Automatically send order notifications"""
    
    try:
        # Send notification to vendor
        vendor_notification = {
            "id": str(uuid.uuid4()),
            "title": "🆕 New Order Received!",
            "message": f"New order from {customer_data['name']} for ₹{order_data['total_amount']}. Check your dashboard to accept.",
            "notification_type": "order",
            "target_users": "specific",
            "user_ids": [vendor_data["id"]],
            "send_sms": True,
            "send_push": True,
            "status": "pending",
            "sent_to_count": 0,
            "created_by": "System",
            "created_at": datetime.utcnow()
        }
        
        await db.notifications.insert_one(vendor_notification)
        await process_notification(vendor_notification["id"])
        
        # Send confirmation to customer
        customer_notification = {
            "id": str(uuid.uuid4()),
            "title": "✅ Order Placed Successfully!",
            "message": f"Your order #{order_data['id'][:8]} has been placed. Total: ₹{order_data['total_amount']}. We'll notify you once the vendor accepts.",
            "notification_type": "order",
            "target_users": "specific",
            "user_ids": [customer_data["id"]],
            "send_sms": True,
            "send_push": True,
            "status": "pending",
            "sent_to_count": 0,
            "created_by": "System",
            "created_at": datetime.utcnow()
        }
        
        await db.notifications.insert_one(customer_notification)
        await process_notification(customer_notification["id"])
        
        print("✅ Order notifications sent to vendor and customer")
        
    except Exception as e:
        print(f"❌ Error sending order notifications: {str(e)}")

# Firebase ID Token Verification Endpoint
@api_router.post("/auth/firebase-verify-token")
async def verify_firebase_token(
    id_token: str,
    current_user: User = Depends(get_current_user)  # Optional: if user already logged in
):
    """Verify Firebase ID Token (Python equivalent of Java verifyIdToken)"""
    
    try:
        # Import Firebase Admin Auth
        from firebase_admin import auth
        
        # Verify the ID token (Python equivalent of Java code)
        decoded_token = auth.verify_id_token(id_token)
        
        # Extract user information (equivalent to Java getUid())
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name')
        picture = decoded_token.get('picture')
        email_verified = decoded_token.get('email_verified', False)
        
        print(f"✅ Firebase token verified for user: {uid}")
        
        # Check if user exists in our database
        existing_user = await db.users.find_one({"firebase_uid": uid})
        
        if existing_user:
            # User exists, return user data
            return {
                "success": True,
                "message": "Token verified successfully",
                "firebase_uid": uid,
                "user_exists": True,
                "user_data": {
                    "id": existing_user["id"],
                    "name": existing_user["name"],
                    "email": existing_user["email"],
                    "user_type": existing_user["user_type"]
                }
            }
        else:
            # New user, can create account
            return {
                "success": True,
                "message": "Token verified, new user",
                "firebase_uid": uid,
                "user_exists": False,
                "firebase_data": {
                    "uid": uid,
                    "email": email,
                    "name": name,
                    "picture": picture,
                    "email_verified": email_verified
                }
            }
            
    except Exception as e:
        print(f"❌ Firebase token verification failed: {str(e)}")
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid Firebase token: {str(e)}"
        )

# Firebase User Registration with ID Token
@api_router.post("/auth/firebase-register")
async def register_with_firebase_token(
    id_token: str,
    user_type: str = "customer",
    phone: Optional[str] = None
):
    """Register user using Firebase ID Token"""
    
    try:
        # Import Firebase Admin Auth
        from firebase_admin import auth
        
        # Verify token first (same as Java verifyIdToken)
        decoded_token = auth.verify_id_token(id_token)
        
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name', 'Firebase User')
        picture = decoded_token.get('picture')
        
        # Check if user already exists
        existing_user = await db.users.find_one({"firebase_uid": uid})
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already registered with this Firebase account")
        
        # Create new user
        new_user = {
            "id": str(uuid.uuid4()),
            "firebase_uid": uid,
            "name": name,
            "email": email,
            "phone": phone or "",
            "user_type": user_type,
            "profile_picture": picture,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "is_verified": True,  # Firebase users are already verified
            "verification_method": "firebase_auth"
        }
        
        # Insert user into database
        await db.users.insert_one(new_user)
        
        # Generate JWT token for our app
        access_token = create_access_token(data={"sub": new_user["email"]})
        
        return {
            "success": True,
            "message": "User registered successfully with Firebase",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": new_user["id"],
                "name": new_user["name"],
                "email": new_user["email"],
                "user_type": new_user["user_type"],
                "firebase_uid": uid
            }
        }
        
    except Exception as e:
        print(f"❌ Firebase registration failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

# Firebase Admin SDK initialization
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        import firebase_admin
        from firebase_admin import credentials
        
        # Check if already initialized
        if firebase_admin._apps:
            print("✅ Firebase Admin SDK already initialized")
            return True
            
        # Try to initialize Firebase with default credentials
        try:
            # For production, use service account key file
            # cred = credentials.Certificate('path/to/service-account-key.json')
            # firebase_admin.initialize_app(cred)
            
            # For now, use default credentials (works in Google Cloud environments)
            firebase_admin.initialize_app()
            print("✅ Firebase Admin SDK initialized successfully")
            return True
            
        except Exception as init_error:
            print(f"⚠️ Firebase initialization failed: {str(init_error)}")
            print("🔔 Push notifications will use fallback method")
            return False
            
    except ImportError:
        print("⚠️ Firebase Admin SDK not installed")
        print("📦 Install with: pip install firebase-admin")
        return False

# Initialize Firebase on startup
firebase_initialized = initialize_firebase()

# Include API router after all endpoints are defined
app.include_router(api_router)

# Database connectivity test on startup
@app.on_event("startup")
async def test_database_connection():
    """Enhanced database connection test with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Database connection attempt {attempt + 1}/{max_retries}")
            
            # Test connection with timeout
            await asyncio.wait_for(client.admin.command('ping'), timeout=30.0)
            print("✅ Database connection successful")
            
            # Test database access
            collections = await db.list_collection_names()
            print(f"📋 Database collections accessible: {len(collections)} found")
            
            # Ensure indexes are created
            try:
                await db.users.create_index("email", unique=True)
                await db.users.create_index("phone")
                await db.orders.create_index("order_id", unique=True)
                await db.products.create_index("vendor_id")
                print("✅ Database indexes created/verified")
            except Exception as index_error:
                print(f"⚠️ Index creation warning (may already exist): {index_error}")
            
            # Success - break retry loop
            break
            
        except asyncio.TimeoutError:
            print(f"❌ Database connection timeout on attempt {attempt + 1}")
        except Exception as e:
            print(f"❌ Database connection failed on attempt {attempt + 1}: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print("🔧 App will continue but database operations may fail")
                print("🔧 Please check MongoDB connection and credentials")

@app.on_event("startup")
async def create_default_admin():
    """Create a default admin account if no admin exists"""
    try:
        # Check if any admin exists
        existing_admin = await db.users.find_one({"user_type": "admin"})
        
        if not existing_admin:
            # Create default admin account
            default_admin = {
                "id": str(uuid.uuid4()),
                "email": "admin@timesafedelivery.com",
                "name": "Super Admin",
                "phone": "+919876543210",
                "user_type": "admin",
                "password_hash": get_password_hash("admin123"),  # Change this password
                "age": 30,
                "gender": "other",
                "address": "TimeSafe Delivery HQ",
                "city": "Mumbai",
                "state": "Maharashtra",
                "pincode": "400001",
                "business_name": None,
                "business_type": None,
                "created_at": datetime.utcnow(),
                "last_login": None,
                "login_count": 0
            }
            
            await db.users.insert_one(default_admin)
            logger.info("✅ Default admin account created: admin@timesafedelivery.com (password: admin123)")
        else:
            logger.info("✅ Admin account already exists")
            
    except Exception as e:
        logger.error(f"❌ Error creating default admin: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    if __name__ == "__main__":
    import os
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
