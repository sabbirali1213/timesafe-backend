#!/usr/bin/env python3
"""
Update TimeSafe Delivery products with beautiful, professional food images
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
import random

# Beautiful food images from professional photography
BEAUTIFUL_FOOD_IMAGES = [
    # Biryani - Traditional and appetizing
    "https://images.unsplash.com/photo-1701579231305-d84d8af9a3fd?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NjZ8MHwxfHNlYXJjaHwxfHxiaXJ5YW5pfGVufDB8fHx8MTc1NTc3MTEwM3ww&ixlib=rb-4.1.0&q=85",
    
    # Butter Chicken - Classic and mouth-watering
    "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzF8MHwxfHNlYXJjaHwyfHxjaGlja2VuJTIwY3Vycnl8ZW58MHx8fHwxNzU1NzcxMTEwfDA&ixlib=rb-4.1.0&q=85",
    
    # Complete Chicken Meal - Perfect for delivery
    "https://images.unsplash.com/photo-1707448829764-9474458021ed?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzF8MHwxfHNlYXJjaHw0fHxjaGlja2VuJTIwY3Vycnl8ZW58MHx8fHwxNzU1NzcxMTEwfDA&ixlib=rb-4.1.0&q=85",
    
    # Rich Mutton Curry - Authentic and appealing
    "https://images.unsplash.com/photo-1596797038530-2c107229654b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzh8MHwxfHNlYXJjaHwzfHxtdXR0b24lMjBjdXJyeXxlbnwwfHx8fDE3NTU3NzExMTl8MA&ixlib=rb-4.1.0&q=85",
    
    # Dhaba-Style Mutton - Complete meal presentation
    "https://images.unsplash.com/photo-1727280213367-ecca818ba188?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzh8MHwxfHNlYXJjaHw0fHxtdXR0b24lMjBjdXJyeXxlbnwwfHx8fDE3NTU3NzExMTl8MA&ixlib=rb-4.1.0&q=85"
]

async def update_product_images():
    """Update all products with beautiful food images"""
    
    # Get MongoDB connection
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_url)
    db = client.timesafe_delivery
    
    print("🍽️ Starting product image update...")
    
    try:
        # Get all products
        products = await db.products.find({}).to_list(length=None)
        print(f"📦 Found {len(products)} products to update")
        
        updated_count = 0
        
        for product in products:
            # Select a random beautiful image
            selected_image = random.choice(BEAUTIFUL_FOOD_IMAGES)
            
            # Create professional image array (3-5 images per product)
            num_images = random.randint(3, 5)
            professional_images = []
            
            # Add the main selected image
            professional_images.append(selected_image)
            
            # Add additional images (can repeat some, like different angles)
            for _ in range(num_images - 1):
                professional_images.append(random.choice(BEAUTIFUL_FOOD_IMAGES))
            
            # Update product with beautiful images
            update_data = {
                "image_url": selected_image,  # Main image
                "image_urls": professional_images  # Multiple images for slideshow
            }
            
            await db.products.update_one(
                {"id": product["id"]},
                {"$set": update_data}
            )
            
            updated_count += 1
            print(f"✅ Updated product: {product['name']} with {len(professional_images)} beautiful images")
        
        print(f"🎉 Successfully updated {updated_count} products with professional food images!")
        print("🤤 Your customers will love these appetizing images!")
        
    except Exception as e:
        print(f"❌ Error updating product images: {e}")
    
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(update_product_images())