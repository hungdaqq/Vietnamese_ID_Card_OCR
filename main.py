from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from extractor import Extractor

idcard_extractor = Extractor()


router = APIRouter()


@router.post("/ocr-idcard")
async def upload_image(file: UploadFile = File(...)):
    # try:
    # Read the image from the uploaded file
    contents = await file.read()
    img_array = np.asarray(bytearray(contents), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Pass the image to the idcard_extractor for detection
    annotations = idcard_extractor.Detection(img)
    # Extract the information from the detected boxes
    # extracted_result = []
    # for i, box in enumerate(reversed(annotations)):
    #     t = idcard_extractor.WarpAndRec(img, box[0][0], box[0][1], box[0][2], box[0][3])
    #     extracted_result.append(t)

    extracted_result = [[annotation[1][0], annotation[0]] for annotation in annotations]

    print(extracted_result)
    # Parse the information from the extracted result
    info = idcard_extractor.GetInformationAndSave(extracted_result)
    # Return the annotations as a response
    return JSONResponse(status_code=200, content={"data": info})


# except Exception as e:
#     # If any error occurs, return an error response
#     return JSONResponse(status_code=500, content={"message": str(e)})


app = FastAPI()

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)
