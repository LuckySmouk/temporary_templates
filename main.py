from openai import AsyncOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate

async def predict_delivery(deal_id: int):
    # 1. Получение данных
    deal = await bitrix.get_deal(deal_id)
    history = await qdrant.similarity_search(deal["params"])
    
    # 2. GPT-прогноз
    client = AsyncOpenAI()
    prompt = ChatPromptTemplate.from_template("""
    Как эксперт логистики GrandTent, предскажи срок доставки:
    Параметры: {params}
    История похожих заказов: {history}
    Правило: Ответ ТОЛЬКО в формате JSON: {{"days": int, "risk": "low|medium|high"}}
    """)
    
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt.format(params=deal, history=history)}]
    )
    
    # 3. Обновление CRM
    forecast = json.loads(response.choices[0].message.content)
    await bitrix.update_deal(deal_id, {"DELIVERY_DAYS": forecast["days"]})
    
    # 4. Уведомление
    if forecast["risk"] == "high":
        await send_alert(f"Риск задержки заказа {deal_id}")
