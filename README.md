import re
import numpy as np

class LinearProgrammingProblem:
    def __init__(self, objective, constraints, maximize=True):
        self.objective = objective
        self.constraints = constraints
        self.maximize = maximize
        
        self.var_names = []
        self.A = None
        self.b = None
        self.c = None
        self.num_vars = 0
        self.num_constraints = 0
        
        self.parse_problem()

    def parse_expression(self, expr):
        # Parse linear expression like "3x + 4y - 2z"
        expr = expr.replace('-', '+-')
        terms = expr.split('+')
        coeffs = {}
        var_names = set()
        for term in terms:
            term = term.strip()
            if term == '':
                continue
            m = re.match(r'([-+]?\d*\.?\d*)\s*([a-zA-Z]\w*)', term)
            if m:
                coef_str, var = m.groups()
                coef = float(coef_str) if coef_str not in ['', '+', '-'] else (1.0 if coef_str in ['', '+'] else -1.0)
                coeffs[var] = coeffs.get(var, 0) + coef
                var_names.add(var)
            else:
                # constant term or error
                try:
                    # constant term without variable
                    val = float(term)
                    coeffs['const'] = coeffs.get('const', 0) + val
                except:
                    raise ValueError(f"No se pudo interpretar el término: {term}")
        return coeffs, var_names

    def parse_problem(self):
        # Extrae todas las variables y crea matrices A, b, c
        
        # Extraemos variables del objetivo y las restricciones
        var_set = set()
        c_dict, obj_vars = self.parse_expression(self.objective)
        var_set.update(obj_vars)
        
        b_list = []
        A_list = []
        self.num_constraints = len(self.constraints)
        
        for constr in self.constraints:
            # Cada restricción en formato "3x + 2y <= 5" o "x + 4y >= 7"
            
            # Detectar operador
            op_search = re.search(r'(<=|>=|=)', constr)
            if not op_search:
                raise ValueError("Cada restricción debe tener un operador (<=, >=, =).")
            op = op_search.group(1)
            
            left_side = constr[:op_search.start()].strip()
            right_side = constr[op_search.end():].strip()
            
            # Parse left side
            coeffs_l, vars_l = self.parse_expression(left_side)
            var_set.update(vars_l)
            # Parse right side should be a constant
            try:
                rhs = float(right_side)
            except:
                raise ValueError(f"Lado derecho no es un número válido: {right_side}")
            
            # Guardar la restricción y el operador para convertir más tarde
            A_list.append((coeffs_l, rhs, op))
        
        self.var_names = sorted(var_set)
        self.num_vars = len(self.var_names)
        
        # Vector c en orden de var_names
        self.c = np.zeros(self.num_vars)
        for i, v in enumerate(self.var_names):
            self.c[i] = c_dict.get(v, 0)
        if not self.maximize:
            self.c = -self.c  # para minimizar se usa -c
        
        # Ahora convertimos restricciones a forma estándar (<=) con variables slack y matrices
        # Contamos cuántas restricciones tipo >= para convertir a <= multiplicando por -1
        # Añadimos variables de holgura
        
        self.A = []
        self.b = []
        self.relation = []  # Guardamos si era <=, =, >= porque el simplex tradicional resuelve <=
        self.slack_var_count = 0
        
        for (coeffs_l, rhs, op) in A_list:
            row = np.zeros(self.num_vars)
            for i, v in enumerate(self.var_names):
                row[i] = coeffs_l.get(v, 0)
            if op == '>=':
                row = -row
                rhs = -rhs
                op = '<='
            self.A.append(row)
            self.b.append(rhs)
            self.relation.append(op)
        self.A = np.array(self.A)
        self.b = np.array(self.b)
        
    def to_canonical(self):
        # Convierte a forma canonica agregando variables de holgura
        # Agrega variables slack para <=
        A_canon = []
        b_canon = []
        slack_count = 0
        
        for i, op in enumerate(self.relation):
            row = list(self.A[i])
            if op == '<=':
                slack = np.zeros(self.num_constraints)
                slack[i] = 1
                row = np.concatenate([row, slack])
                slack_count += 1
            else:  # op == '='
                # igualdad, no agregamos slack
                slack = np.zeros(self.num_constraints)
                row = np.concatenate([row, slack])
            A_canon.append(row)
        
        self.A = np.array(A_canon)
        self.b = np.array(self.b)
        
        self.num_vars = self.A.shape[1]
        
        # Extender c con ceros para variables slack
        self.c = np.concatenate([self.c, np.zeros(self.num_constraints)])
    
    def simplex(self):
        # Resuelve el problema con el método simplex
        # Método simplex para maximizar c^T x sujeto a Ax <= b, x >= 0
        # Usa tabla simplex con variables básicas y no básicas
        
        m, n = self.A.shape
        
        # Tabla simplex: filas m+1, columnas n+1 (última columna b)
        # Tabla inicial
        table = np.zeros((m+1, n+1))
        table[:-1, :-1] = self.A
        table[:-1, -1] = self.b
        table[-1, :-1] = -self.c
        
        # Variables básicas son las variables slack al inicio
        basis = list(range(n - m, n))
        
        def pivot(r, c):
            # pivotea en tabla
            table[r, :] = table[r, :] / table[r, c]
            for i in range(m+1):
                if i != r:
                    table[i, :] = table[i, :] - table[i, c] * table[r, :]
            basis[r] = c
        
        iteration = 0
        max_iterations = 100
        
        history = []
        
        while True:
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("Máximo de iteraciones alcanzado.")
            
            history.append(table.copy())
            
            # criterio de optimalidad: coeficientes negativos en fila última
            c_row = table[-1, :-1]
            if all(c_row >= 0):
                break  # óptimo
            
            # variable entrante: el índice del coeficiente negativo más negativo (regla Bland diferente)
            c_index = np.argmin(c_row)
            
            # proporción para variable saliente
            ratios = []
            for i in range(m):
                a_ij = table[i, c_index]
                if a_ij > 0:
                    ratios.append(table[i, -1] / a_ij)
                else:
                    ratios.append(np.inf)
            
            if all(r == np.inf for r in ratios):
                raise RuntimeError("Solución no acotada.")
            
            r_index = np.argmin(ratios)
            pivot(r_index, c_index)
        
        # Extraer solución
        x = np.zeros(n)
        for i, var_index in enumerate(basis):
            if var_index < n:
                x[var_index] = table[i, -1]
        
        value = table[-1, -1]
        if not self.maximize:
            value = -value
        
        return x[:len(self.var_names)], value, history

    def interpret_solution(self, x, optimum):
        res = f"Solución óptima:\n"
        for i, v in enumerate(self.var_names):
            res += f"  {v} = {x[i]:.4f}\n"
        res += f"Valor óptimo de la función objetivo: {optimum:.4f}\n"
        return res

    def show_simplex_steps(self, history):
        step_str = ""
        for i, table in enumerate(history):
            step_str += f"\nIteración {i}:\n"
            step_str += np.array2string(table, formatter={'float_kind':lambda x: f"{x:8.4f}"}) + "\n"
        return step_str

def main():
    print("Programa para resolver problemas de Programación Lineal (método Simplex)")
    print("Ingrese la función objetivo, ejemplo: max 3x + 2y - z")
    line = input("¿Maximizar o Minimizar? (max/min): ").strip().lower()
    maximize = True if line == 'max' else False
    
    obj = input("Ingrese la función objetivo (ejemplo: 3x + 2y - z): ").strip()
    
    print("Ingrese las restricciones, una por línea, en formato por ejemplo:")
    print("3x + 2y <= 5")
    print("x - y >= 1")
    print("Para terminar ingrese línea vacía.")
    
    constraints = []
    while True:
        c = input("Restricción: ").strip()
        if c == '':
            break
        constraints.append(c)
    
    try:
        lp = LinearProgrammingProblem(obj, constraints, maximize=maximize)
        lp.to_canonical()
        solution, optimum, history = lp.simplex()
        print(lp.interpret_solution(solution, optimum))
        print("Proceso Simplex paso a paso:")
        print(lp.show_simplex_steps(history))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

