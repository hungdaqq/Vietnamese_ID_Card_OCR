from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time  # Import time module to measure execution time
from extractor import Extractor

idcard_extractor = Extractor()

router = APIRouter()


@router.post("/ocr")
async def upload_image(
    id_card_front: UploadFile = File(...),
    id_card_back: UploadFile = File(...),
):
    try:
        # Start timing for the entire request
        start_time = time.time()

        # Read the front image from the uploaded file
        front_start_time = time.time()
        front = await id_card_front.read()
        front_array = np.asarray(bytearray(front), dtype=np.uint8)
        front_img = cv2.imdecode(front_array, cv2.IMREAD_COLOR)
        front_end_time = time.time()

        # Read the back image from the uploaded file
        back_start_time = time.time()
        back = await id_card_back.read()
        back_array = np.asarray(bytearray(back), dtype=np.uint8)
        back_img = cv2.imdecode(back_array, cv2.IMREAD_COLOR)
        back_end_time = time.time()

        # Pass the image to the idcard_extractor for detection
        front_detection_start_time = time.time()
        front_annotations = idcard_extractor.Detection(front_img)
        front_detection_end_time = time.time()

        back_detection_start_time = time.time()
        back_annotations = idcard_extractor.Detection(back_img)
        back_detection_end_time = time.time()

        # Extract the information from the detected boxes for the front
        front_extract_start_time = time.time()
        extracted_result = []
        for _, box in enumerate(reversed(front_annotations)):
            t = idcard_extractor.WarpAndRec(front_img, box[0], box[1], box[2], box[3])
            extracted_result.append(t)
        front_extract_end_time = time.time()

        # Parse the information from the extracted result for the front
        front_info_start_time = time.time()
        front_info = idcard_extractor.GetInformationFront(extracted_result)
        front_info_end_time = time.time()

        # Extract the information from the detected boxes for the back
        back_extract_start_time = time.time()
        extracted_result = []
        for _, box in enumerate(reversed(back_annotations)):
            t = idcard_extractor.WarpAndRec(back_img, box[0], box[1], box[2], box[3])
            extracted_result.append(t)
        back_extract_end_time = time.time()

        # Parse the information from the extracted result for the back
        back_info_start_time = time.time()
        back_info = idcard_extractor.GetInformationBack(extracted_result)
        back_info_end_time = time.time()

        # End timing for the entire request
        end_time = time.time()

        # Log the times for each stage
        print(f"Front Image Read Time: {front_end_time - front_start_time:.4f} seconds")
        print(f"Back Image Read Time: {back_end_time - back_start_time:.4f} seconds")
        print(
            f"Front Detection Time: {front_detection_end_time - front_detection_start_time:.4f} seconds"
        )
        print(
            f"Back Detection Time: {back_detection_end_time - back_detection_start_time:.4f} seconds"
        )
        print(
            f"Front Extraction Time: {front_extract_end_time - front_extract_start_time:.4f} seconds"
        )
        print(
            f"Front Info Parsing Time: {front_info_end_time - front_info_start_time:.4f} seconds"
        )
        print(
            f"Back Extraction Time: {back_extract_end_time - back_extract_start_time:.4f} seconds"
        )
        print(
            f"Back Info Parsing Time: {back_info_end_time - back_info_start_time:.4f} seconds"
        )
        print(f"Total Request Processing Time: {end_time - start_time:.4f} seconds")

        # Return the annotations as a response
        return JSONResponse(
            status_code=200,
            content={
                "status_code": 200,
                "message": "Trích xuất thông tin CCCD thành công",
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
                "message": "Không thể trích xuất thông tin CCCD",
                "data": None,
                "error": str(e),
            },
        )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)
