from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from extractor import Extractor

idcard_extractor = Extractor()


router = APIRouter()


@router.post("/ocr")
async def upload_image(
    id_card_front: UploadFile = File(...),
    id_card_back: UploadFile = File(...),
):
    try:
        # Read the front image from the uploaded file
        front = await id_card_front.read()
        front_array = np.asarray(bytearray(front), dtype=np.uint8)
        front_img = cv2.imdecode(front_array, cv2.IMREAD_COLOR)

        # Read the back image from the uploaded file
        back = await id_card_back.read()
        back_array = np.asarray(bytearray(back), dtype=np.uint8)
        back_img = cv2.imdecode(back_array, cv2.IMREAD_COLOR)

        # Pass the image to the idcard_extractor for detection
        front_annotations = idcard_extractor.Detection(front_img)
        back_annotations = idcard_extractor.Detection(back_img)

        # Extract the information from the detected boxes
        extracted_result = []
        for _, box in enumerate(reversed(front_annotations)):
            t = idcard_extractor.WarpAndRec(
                # front_img, box[0][0], box[0][1], box[0][2], box[0][3]
                front_img,
                box[0],
                box[1],
                box[2],
                box[3],
            )
            extracted_result.append(t)

        # Parse the information from the extracted result
        front_info = idcard_extractor.GetInformationFront(extracted_result)

        # Extract the information from the detected boxes
        extracted_result = []
        for _, box in enumerate(reversed(back_annotations)):
            t = idcard_extractor.WarpAndRec(
                # front_img, box[0][0], box[0][1], box[0][2], box[0][3]
                back_img,
                box[0],
                box[1],
                box[2],
                box[3],
            )
            extracted_result.append(t)

        # Parse the information from the extracted result
        back_info = idcard_extractor.GetInformationBack(extracted_result)

        # Return the annotations as a response
        return JSONResponse(
            status_code=200,
            content={
                "status_code": 200,
                "message": "Successfully extracted the information",
                "data": {
                    "identity_card_number": front_info["identity_card_number"],
                    "full_name": front_info["full_name"],
                    "date_of_birth": front_info["date_of_birth"],
                    "gender": front_info["gender"],
                    "nationality": front_info["nationality"],
                    "place_of_origin": front_info["place_of_origin"],
                    "place_of_residence": front_info["place_of_residence"],
                    "id_card_issued_date": back_info["id_card_issued_date"],
                    "id_card_expired_date": front_info["id_card_expired_date"],
                },
                "error": None,
            },
        )

    except Exception as e:
        # If any error occurs, return an error response
        return JSONResponse(
            status_code=400,
            content={
                "status_code": 400,
                "message": "Failed to extract the information",
                "data": None,
                "error": str(e),
            },
        )


app = FastAPI()

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)
