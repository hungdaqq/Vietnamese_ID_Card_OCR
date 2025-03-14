from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from extractor import Extractor
import ultralytics
import utils
import requests

idcard_extractor = Extractor()

router = APIRouter()

model = ultralytics.YOLO("best.pt")


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

        back = await id_card_back.read()
        back_array = np.asarray(bytearray(back), dtype=np.uint8)
        back_img = cv2.imdecode(back_array, cv2.IMREAD_COLOR)

        results = model([front_img, back_img])
        images = []
        for img, result in zip([front_img, back_img], results):
            points = {}
            # Iterate over detected objects
            for box in result.boxes:
                # Get the class index and coordinates
                class_index = int(box.cls.item())  # Convert tensor to int
                x1, y1, x2, y2 = box.xyxy[0]  # Assuming xyxy format

                # Calculate the center of the box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Map the class index to a corner name
                if class_index in utils.class_to_corner:
                    corner_name = utils.class_to_corner[class_index]
                    points[corner_name] = (center_x, center_y)

            if all(
                k in points
                for k in ["top_left", "top_right", "bottom_right", "bottom_left"]
            ):
                rect = np.array(
                    [
                        points["top_left"],
                        points["top_right"],
                        points["bottom_right"],
                        points["bottom_left"],
                    ],
                    dtype="float32",
                )

                width = int(np.linalg.norm(rect[0] - rect[1]))
                height = int(np.linalg.norm(rect[0] - rect[3]))
                dst = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype="float32",
                )

                M = cv2.getPerspectiveTransform(rect, dst)
                img = cv2.warpPerspective(img, M, (width, height))

            images.append(img)

        # Front
        front_annotations = idcard_extractor.Detection(images[0])
        back_annotations = idcard_extractor.Detection(images[1])
        extracted_result = []
        for _, box in enumerate(reversed(front_annotations)):
            t = idcard_extractor.WarpAndRec(images[0], box[0], box[1], box[2], box[3])
            extracted_result.append(t)
        front_info = idcard_extractor.GetInformationFront(extracted_result)
        # Back
        back_annotations = idcard_extractor.Detection(images[1])
        extracted_result = []
        for _, box in enumerate(reversed(back_annotations)):
            t = idcard_extractor.WarpAndRec(images[1], box[0], box[1], box[2], box[3])
            extracted_result.append(t)
        back_info = idcard_extractor.GetInformationBack(extracted_result)

        front_dir = f"./tmp/{front_info['identity_card_number']}_mattruoc.jpg"
        back_dir = f"./tmp/{front_info['identity_card_number']}_matsau.jpg"
        cv2.imwrite(front_dir, images[0])
        cv2.imwrite(back_dir, images[1])

        
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
                    "id_card_front": id_card_front,
                    "id_card_back": id_card_back,
                },
                "error": None,
            },
        )

    except Exception as e:
        # If any error occurs, return an error response
        print(e)
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
