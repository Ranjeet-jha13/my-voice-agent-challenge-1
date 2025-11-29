import json
import os
from datetime import datetime

class MerchantAPI:
    def __init__(self):
        # Smart Path Finding (Fixes the "File Not Found" errors from previous days)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.catalog_path = os.path.join(base_dir, "products.json")
        self.orders_path = os.path.join(base_dir, "orders.json")
        self.products = self._load_catalog()

    def _load_catalog(self):
        try:
            with open(self.catalog_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ Error: products.json not found.")
            return []

    def search_products(self, query=None, category=None, max_price=None):
        """Filters products based on criteria with smarter keyword matching."""
        results = self.products
        
        # 1. Filter by Category
        if category:
            results = [p for p in results if category.lower() in p['category'].lower()]
        
        # 2. Filter by Price
        if max_price:
            try:
                # Handle if AI passes "$100" or "100"
                price_val = float(str(max_price).replace('$', '').replace(',', ''))
                results = [p for p in results if p['price'] <= price_val]
            except ValueError:
                pass # Ignore invalid price filters
            
        # 3. Smart Keyword Search (The Fix!)
        if query:
            q_terms = query.lower().split() # Split "black hoodie" -> ["black", "hoodie"]
            filtered = []
            
            for p in results:
                # Create a "Searchable Text" blob for this product
                # It includes Name, Category, and all Attributes values
                # e.g. "developer hoodie clothing black cotton..."
                searchable_text = (
                    p['name'].lower() + " " + 
                    p['category'].lower() + " " + 
                    " ".join(str(v).lower() for v in p.get('attributes', {}).values())
                )
                
                # Check if ALL query words exist in the product data
                # If user says "Black Hoodie", both "black" and "hoodie" must be in the text
                if all(term in searchable_text for term in q_terms):
                    filtered.append(p)
            
            results = filtered
            
        return results
    def create_order(self, product_name, quantity, customer_name):
        """Creates an order object and saves it."""
        # 1. Find the product
        product = next((p for p in self.products if product_name.lower() in p['name'].lower()), None)
        
        if not product:
            return {"error": f"Product '{product_name}' not found."}

        # 2. Construct Order Object (ACP Style)
        total_price = product['price'] * quantity
        order = {
            "order_id": f"ORD-{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer": customer_name,
            "items": [
                {
                    "product_id": product['id'],
                    "name": product['name'],
                    "quantity": quantity,
                    "unit_price": product['price']
                }
            ],
            "total_amount": total_price,
            "currency": product['currency'],
            "status": "confirmed"
        }

        # 3. Persist to JSON
        existing_orders = []
        if os.path.exists(self.orders_path):
            try:
                with open(self.orders_path, "r") as f:
                    existing_orders = json.load(f)
            except: pass
            
        existing_orders.append(order)
        
        with open(self.orders_path, "w") as f:
            json.dump(existing_orders, f, indent=2)
            
        return order

    def get_last_order(self):
        """Retrieves the most recent order."""
        if not os.path.exists(self.orders_path):
            return None
        try:
            with open(self.orders_path, "r") as f:
                orders = json.load(f)
                return orders[-1] if orders else None
        except:
            return None