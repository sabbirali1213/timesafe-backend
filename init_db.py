#!/usr/bin/env python3
"""
Database initialization script for production deployment
Handles Atlas MongoDB connection and creates default admin user
"""

import os
import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__default_rounds=12)

async def init_database():
    """Initialize database with proper Atlas connection"""
    
    # Get environment variables
    mongo_url = os.environ.get('MONGO_URL')
    db_name = os.environ.get('DB_NAME', 'timesafe_delivery')
    
    if not mongo_url:
        print("❌ MONGO_URL environment variable not set")
        return False
        
    print(f"🔧 Connecting to database: {db_name}")
    print(f"🔗 MongoDB URL: {mongo_url[:30]}...")
    
    try:
        # Atlas-compatible connection options
        connection_options = {
            'serverSelectionTimeoutMS': 30000,
            'connectTimeoutMS': 30000,
            'maxPoolSize': 10,
            'retryWrites': True
        }
        
        # Add authentication for Atlas
        if '@' in mongo_url and ('mongodb+srv://' in mongo_url or 'mongodb://' in mongo_url):
            connection_options.update({
                'authSource': 'admin',
                'authMechanism': 'SCRAM-SHA-1'
            })
            
        client = AsyncIOMotorClient(mongo_url, **connection_options)
        db = client[db_name]
        
        # Test connection
        await client.admin.command('ping')
        print("✅ Database connection successful")
        
        # Create indexes - PREVENT DUPLICATE REGISTRATIONS
        try:
            await db.users.create_index("email", unique=True)
            await db.users.create_index("phone", unique=True)  # ✅ UNIQUE PHONE NUMBERS
            # Create compound index on both id and order_id for compatibility
            await db.orders.create_index("order_id", unique=True)
            await db.orders.create_index("id", unique=True) 
            await db.orders.create_index([("customer_id", 1), ("created_at", -1)])  # For faster customer order queries
            await db.products.create_index("vendor_id")
            print("✅ Database indexes created with unique phone constraint")
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
        
        # Create default admin user if not exists
        admin_exists = await db.users.find_one({"email": "admin@timesafe.in"})
        if not admin_exists:
            admin_user = {
                "user_id": "admin-001",
                "name": "TimeSafe Admin",
                "email": "admin@timesafe.in",
                "phone": "+911234567890",
                "password_hash": pwd_context.hash("admin123"),
                "user_type": "admin",
                "is_verified": True,
                "created_at": datetime.utcnow(),
                "last_login": None,
                "login_count": 0
            }
            
            await db.users.insert_one(admin_user)
            print("✅ Default admin user created: admin@timesafe.in / admin123")
        else:
            print("✅ Admin user already exists")
            
        await client.close()
        print("🎉 Database initialization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting database initialization...")
    
    success = asyncio.run(init_database())
    
    if success:
        print("✅ Database initialization completed")
        sys.exit(0)
    else:
        print("❌ Database initialization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()