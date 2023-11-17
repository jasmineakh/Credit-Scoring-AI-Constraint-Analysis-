import numpy as np
from flask import Flask, request, jsonify
import pickle 

app = Flask(__name__)
model = pickle.load(open('model/GBR_pickle.pkl','rb'))
 
@app.route('/api',methods=['POST'])
def predict():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            params = request.json
            input_data = [x for x in params.values()]

            final_features = [np.array(input_data)]

            result = model.predict(final_features)[0]

            # criteria
            if result < 34:
                criteria = 'low'
            elif result >= 34 and result < 67:
                criteria = 'medium'
            else:
                criteria = 'high'
                
            return jsonify({
                'status': True,
                'message': 'success',
                'input': params,
                'result': result, 
                'criteria': criteria
            })
        
    except Exception as e:
        return jsonify({
                'status': False,
                'message': str(e)
            })

if __name__ =='__main__':
    app.run(port=5000, debug=True)

