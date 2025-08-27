import os

# MongoDB Atlas Configuration
MONGODB_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
DATABASE_NAME = "timesafe_db"
CONNECTION_OPTIONS = {
    "maxPoolSize": 50,
    "minPoolSize": 5,
    "maxIdleTimeMS": 30000,
    "serverSelectionTimeoutMS": 5000,
}
