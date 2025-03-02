import database.db as DB
from database.connection import get_db
from agents.assistant import ChatMessage, get_chat_progress
import asyncio


async def main():
    db = next(get_db())

    user = DB.get_user(db, 1)
    situation = DB.get_situation(db, 4)

    messages = [
        ChatMessage(role="human", content="Hola"),
        ChatMessage(role="ai", content="¡Hola! Bienvenido al hotel. ¿Cómo se llama?"),
        ChatMessage(role="human", content="Juanito de la Vera"),
        ChatMessage(
            role="ai",
            content="Encantado, Juanito. ¿Tiene su reserva lista? Necesitamos verla para continuar con el check-in.",
        ),
        ChatMessage(role="human", content="si la tengo aqui en mi celular"),
        ChatMessage(
            role="ai",
            content="Perfecto. Por favor, muestre la reserva en su celular. Le ayudaré a completar el proceso de check-in. ¿Dónde está su equipaje?",
        ),
        ChatMessage(role="human", content="tengo solo una mochila chica"),
        ChatMessage(
            role="ai",
            content="Muy bien, Juanito. Su habitación está lista. ¿Quiere que le ayude con la mochila mientras se registra?",
        ),
        ChatMessage(
            role="human",
            content="no hay problema, no es pesada en lo absoluto. Me gustaría saber si el desayuno esta incluido",
        ),
        ChatMessage(
            role="ai",
            content="Sí, el desayuno está incluido. Es de 7 a 10 de la mañana en la cafetería del hotel. ¿Le gusta desayunar temprano?",
        ),
        ChatMessage(role="human", content="a eso de las 8 estaría bien"),
        ChatMessage(
            role="ai",
            content="Perfecto. Puede ir a la cafetería a las 8. También tenemos un restaurante para comer y cenar. ¿Necesita más información sobre los servicios del hotel?",
        ),
        ChatMessage(
            role="human", content="si, hay acaso algún tipo de gimnasio aquì? "
        ),
        ChatMessage(
            role="ai",
            content="Sí, tenemos un pequeño gimnasio en el segundo piso. Está abierto las 24 horas. ¿Le gustaría verlo ahora o más tarde?",
        ),
        ChatMessage(role="human", content="más tarde sin problemas "),
        ChatMessage(
            role="ai",
            content="De acuerdo. En su habitación encontrará información sobre cómo usar la televisión y el aire acondicionado. ¿Necesita ayuda con algo más en este momento?",
        ),
        ChatMessage(
            role="human",
            content="hay algún tipo de transporte público en la cercanìa? ",
        ),
        ChatMessage(
            role="ai",
            content="Sí, hay autobuses cerca del hotel. Le puedo dar un mapa si lo necesita. ¿Le gustaría verlo ahora?",
        ),
        ChatMessage(
            role="human", content="no lo puedo ver en mi celular sin problemas"
        ),
        ChatMessage(
            role="ai",
            content="Entiendo. Puede encontrar el mapa en la recepción cuando quiera. Si tiene alguna pregunta, llámenos al 123-4567. ¿Hay algo más que pueda hacer por usted hoy?",
        ),
        ChatMessage(role="human", content="no muchas gracias"),
        ChatMessage(
            role="ai",
            content="De nada, Juanito. ¡Que pase un buen día! Llame si necesita algo.",
        ),
    ]

    result = await get_chat_progress(user, situation, messages)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
