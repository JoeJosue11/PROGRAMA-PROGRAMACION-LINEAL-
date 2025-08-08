import re
import numpy as np

class NLPLinearProblemExtractor:
    def __init__(self, text):
        self.text = text.lower()
        self.variables = ['x', 'y']  # suponemos dos variables para el ejemplo camiones tipo 1 y 2
        self.data = {}
        self.parse_text()

    def parse_text(self):
        # Ejemplo para problema tipos de camiones
        # Extrae volúmenes de productos refrigerados y no refrigerados
        match_refrigerated = re.search(r'(\d+\.?\d*)\s*m³\s*de producto que necesita refrigeración', self.text)
        match_non_refrigerated = re.search(r'(\d+\.?\d*)\s*m³\s*de otro que no la necesita', self.text)
        match_cost_type1 = re.search(r'coste por kilómetro.*camión del tipo\s*\w+\s*es de\s*(\d+\.?\d*)\s*€', self.text)
        match_cost_type2 = re.findall(r'coste por kilómetro.*?camión del tipo\s*\w+\s*es de\s*(\d+\.?\d*)\s*€', self.text)
        # Espacio refrigerado y no refrigerado tipo 1
        match_space_type1 = re.search(r'tipo\s*1.*?espacio refrigerado de\s*(\d+\.?\d*)\s*m³.*?espacio no refrigerado de\s*(\d+\.?\d*)\s*m³', self.text)
        # Espacio tipo 2 (cubicaje total igual, al % refrigerado)
        match_space_type2_percent = re.search(r'tipo\s*2.*?al\s*(\d+\.?\d*)\s*%.*?refrigerado', self.text)

        # Guardar datos según patrones encontrados
        if match_refrigerated:
            self.data['prod_refrigerated'] = float(match_refrigerated.group(1))
        if match_non_refrigerated:
            self.data['prod_non_refrigerated'] = float(match_non_refrigerated.group(1))
        if match_space_type1:
            self.data['space_refrigerated_1'] = float(match_space_type1.group(1))
            self.data['space_non_refrigerated_1'] = float(match_space_type1.group(2))
            self.data['total_space_1'] = self.data['space_refrigerated_1'] + self.data['space_non_refrigerated_1']
        if match_space_type2_percent:
            pct_refrigerated = float(match_space_type2_percent.group(1))
            self.data['pct_refrigerated_2'] = pct_refrigerated / 100.0
            # Tipo 2 tiene mismo cubicaje total que tipo 1
            if 'total_space_1' in self.data:
                self.data['space_refrigerated_2'] = self.data['total_space_1'] * self.data['pct_refrigerated_2']
                self.data['space_non_refrigerated_2'] = self.data['total_space_1'] * (1 - self.data['pct_refrigerated_2'])
        if match_cost_type1 and len(match_cost_type2) > 1:
            self.data['cost_1'] = float(match_cost_type1.group(1))
            self.data['cost_2'] = float(match_cost_type2[1])
        elif len(match_cost_type2) == 1:
            self.data['cost_1'] = float(match_cost_type2[0])
            self.data['cost_2'] = float(match_cost_type2[0]) # por defecto mismo costo si no hay dos valores

    def build_lp(self):
        # Variables x (camiones tipo 1), y (tipo 2)
        # Función objetivo: minimizar coste total = cost_1*x + cost_2*y
        c = np.array([self.data.get('cost_1', 0), self.data.get('cost_2',0)])
        # Restricciones:
        # Espacio refrigerado necesario: space_refrigerated_1*x + space_refrigerated_2*y >= prod_refrigerated
        # Espacio no refrigerado necesario: space_non_refrigerated_1*x + space_non_refrigerated_2*y >= prod_non_refrigerated
        A = np.array([
            [-self.data.get('space_refrigerated_1',0), -self.data.get('space_refrigerated_2',0)],  # <= -prod_refrigerated
            [-self.data.get('space_non_refrigerated_1',0), -self.data.get('space_non_refrigerated_2',0)]
        ])
        b = np.array([-self.data.get('prod_refrigerated',0), -self.data.get('prod_non_refrigerated',0)])
        return c, A, b

def simplex(c, A, b, maximize=False):
    # Método simplex simple para resolver min c^T x s.a. Ax <= b, x>=0
    # Adaptación del código anterior para resolver con numpy
    if maximize:
        c = -c
    m, n = A.shape
    # Construir tabla simplex con holguras para <=
    A_slack = np.hstack([A, np.eye(m)])
    c_extended = np.concatenate([c, np.zeros(m)])
    
    table = np.zeros((m+1, n + m +1))
    table[:-1, :-1] = A_slack
    table[:-1, -1] = b
    table[-1, :-1] = -c_extended
    
    basis = list(range(n, n+m))
    
    def pivot(r,c):
        table[r, :] = table[r, :]/table[r, c]
        for i in range(m+1):
            if i!=r:
                table[i,:] = table[i,:] - table[i,c]*table[r,:]
        basis[r] = c
    
    iteration = 0
    max_iter = 100
    while True:
        iteration += 1
        if iteration > max_iter:
            raise Exception("Iteraciones maximas alcanzadas.")
        c_row = table[-1,:-1]
        if np.all(c_row >= 0):
            break
        c_index = np.argmin(c_row)
        ratios = []
        for i in range(m):
            if table[i,c_index] > 0:
                ratios.append(table[i,-1]/table[i,c_index])
            else:
                ratios.append(np.inf)
        if all(r==np.inf for r in ratios):
            raise Exception("Solucion no acotada.")
        r_index = np.argmin(ratios)
        pivot(r_index, c_index)
    
    x = np.zeros(n+m)
    for i, var_index in enumerate(basis):
        if var_index < n+m:
            x[var_index] = table[i,-1]
    value = table[-1,-1]
    if maximize:
        value = -value
    return x[:n], value

def main():
    print("Por favor, pega el problema completo en texto (ejemplo camiones), y escribe 'FIN' en una línea vacía para procesar:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'FIN' or line.strip() == '':
            break
        lines.append(line)
    text = ' '.join(lines)
    
    extractor = NLPLinearProblemExtractor(text)
    print("Datos extraídos del texto:")
    for k,v in extractor.data.items():
        print(f"  {k}: {v}")
        
    c, A, b = extractor.build_lp()
    print("\nModelo LP formado:")
    print("Función objetivo (costes):", c)
    print("Restricciones (A x <= b):")
    print("A = ", A)
    print("b = ", b)
    
    solution, optimum = simplex(c, A, b, maximize=False)
    
    print("\nSolución óptima:")
    for i, val in enumerate(solution):
        print(f"  {extractor.variables[i]} = {val:.4f}")
    print(f"Costo mínimo total: {optimum:.4f}")

if __name__ == "__main__":
    main()
