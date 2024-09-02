from langchain.tools import tool
from database.db import get_database
from datetime import datetime
from bson.objectid import ObjectId
from bson.errors import InvalidId

def get_client(customer_id):
    db = get_database()
    try:
        customer = db['client'].find_one({"_id": ObjectId(customer_id)})
    except InvalidId:
        return f"ID cliente non valido: {customer_id}"
    if customer:
        return customer
    else:
        return f"Nessun cliente trovato con ID {customer_id}"

@tool
def get_customer_info(customer_id: str) -> str:
    """Ritorna le informazioni del cliente in base al suo ID cliente."""
    try:
        print(f"\nAttivazione tool get_customer_info con id {customer_id}")
        return get_client(customer_id)
    except Exception as e:
        return "Errore interno del server"

@tool
def get_customer_bills(customer_id: str) -> str:
    """Ritorna la lista delle fatture di un cliente in base al suo ID cliente."""
    try:
        print(f"\nAttivazione tool get_customer_bills con id {customer_id}")
        return get_client(customer_id)
    except Exception as e:
        return "Errore interno del server"
    
@tool
def sign_contract(customer_id: str, contract_type: str):
    """Sottocrivi il contratto che il cliente specifica in base al suo ID cliente."""
    try:
        print(f"\nAttivazione tool sign_contract con id {customer_id}")
        db = get_database()
        activation_time = datetime.now()
        result = db['client'].update_one(
            {"_id": ObjectId(customer_id)},
            {"$set": {"contratto": {"tipo": contract_type, "data_attivazione": activation_time}}}
        )
        if result.modified_count > 0:
            return "Contratto attivato con successo"
        else:
            return f"Impossibile attivare il contratto per il cliente con ID {customer_id}"
    except InvalidId:
        return f"ID cliente non valido: {customer_id}"
    except Exception as e:
        return f"Errore interno del server: {str(e)}"
