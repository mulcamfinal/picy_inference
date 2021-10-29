from flask import Flask
from flask_restful import Resource, reqparse, Api
import werkzeug


class ProcessImageEndpoint(Resource):

    def __init__(self):
        # Create a request parser
        parser = reqparse.RequestParser()
        parser.add_argument(
            "image", type=werkzeug.datastructures.FileStorage, location='files')
        # Sending more info in the form? simply add additional arguments, with the location being 'form'
        # parser.add_argument("other_arg", type=str, location='form')
        self.req_parser = parser

    # This method is called when we send a POST request to this endpoint
    def post(self):
        # The image is retrieved as a file
        image_file = self.req_parser.parse_args(strict=True).get("image", None)
        if image_file:
            # Get the byte content using `.read()`
            image = image_file.read()
            # Now do something with the image...
            return "Yay, you sent an image!"
        else:
            return "No image sent :("


app = Flask(__name__)
api = Api(app)


class CreateUser(Resource):
    def post(self):
        return {'status': 'success'}


api.add_resource(CreateUser, '/user')

if __name__ == '__main__':
    app.run(debug=True)
