import os 
import sys 
import numpy as np 
from utils import DataPoint, ScorerStepByStep
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

class PredictionModel:
    """
    GRU со специальным вниманием к первым 100 наблюдениям.
    """

    def __init__(self, model_path=""):
        """
        Задаёт путь к модели и создаёт ONNX сессию с моделью.
        """
        self.current_seq_ix = None
        self.sequence_history = []
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "my_model.onnx") 
        
        sess_options = ort.SessionOptions()
        # Ставим 1, потому что у нас только 1 CPU core по условию
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = None
        
        try:
            self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name 
            self.output_name = self.ort_session.get_outputs()[0].name 
            print(f"Загружена ONNX модель из {onnx_path}")
                
        except Exception as e:
            print(f"Ошибка загрузки ресурсов модели: {e}")
            self.ort_session = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Делает предсказание на основе текущей точки данных и истории последовательности.
        
        Для каждой новой последовательности сбрасывает историю.
        При необходимости делать предсказание формирует окно из последних 100 наблюдений,
        дополняя нулями, если наблюдений меньше 100.
        """
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None
        if self.ort_session is None:
            return np.zeros(2)

        
        history_window = self.sequence_history[-100:] 
        
        if len(history_window) < 100:
             padding = [np.zeros_like(history_window[0])] * (100 - len(history_window))
             history_window = padding + history_window

        data_arr = np.asarray(history_window, dtype=np.float32)
        data_tensor = np.expand_dims(data_arr, axis=0)
        
        ort_inputs = {self.input_name: data_tensor}
        output = self.ort_session.run([self.output_name], ort_inputs)[0]
        
        if len(output.shape) == 3:
            # если модель возвращает (Batch, Seq, Features)
            prediction = output[0, -1, :]
        else:
            # иначе (Batch, Features)
            prediction = output[0]
            
        return prediction


if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../datasets/valid.parquet"
    
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Тестирование Vanilla GRU Baseline (ONNX)...")
        results = scorer.score(model)
        
        print("\nРезультаты:")
        print(f"Средневзвешенная корреляция Пирсона: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet не найден для тестирования.")
