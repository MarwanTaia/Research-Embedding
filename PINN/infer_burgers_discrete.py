'''
FICHIER:
--------
    infer_burgers_discrete.py

DESCRIPTION:
------------
Ce fichier contient le programme de réimplémentation de PINN (Physics Informed
    Neural Network) pour la résolution de l'équation de Burgers discrète.
On discrétise l'équation de Burgers en utilisant la méthode Runge-Kutta de
    q-ème ordre.
Dans notre cas x est la position et t le temps. Chacun des deux est un vecteur,
    donc unidimensionnel. Le problème est défini sur l'intervalle [0,1] pour x,
    et [0,1] pour t.
La condition initiale est u(x,0) = -sin(pi*x) et la condition aux limites est
    u(0,()) = u(1,t) = 0.
'''

################################################################################
# IMPORTS
################################################################################
import numpy as np 
import tensorflow as tf
import scipy.io as sio # Pour charger les données
from pyDOE import lhs # Latin Hypercube Sampling
from scipy.interpolate import griddata # Interpolation
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution()

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

################################################################################
# MACROS
################################################################################
# Chemin relatif vers le fichier de données
DATA_PATH = 'data/burgers_shock.mat'

# Macros pour les affichages
VERIF_PLOTS = False
PLOTS = True

# Macros pour l'apprentissage
NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.001

################################################################################
# CLASSES
################################################################################
class PINN:
    '''
    Classe PINN
    -----------
    Cette classe contient les méthodes permettant de construire et d'entraîner
        un PINN pour la résolution de l'équation de Burgers.
    
    Dans cette classe, on définit les méthodes:
    - __init__

    Note : dans cette classe, on créé le réseau de neurones directement dans
        la méthode make_NN, et on l'entraîne dans la méthode train. Tout
        directement avec TensorFlow.
    '''

    # def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, nu, q):
        '''
        Méthode __init__
        ----------------
        Cette méthode initialise les attributs de la classe PINN.
        '''

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)       
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('data/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
        self.IRK_alpha = weights[0:-1,:]
        self.IRK_beta = weights[-1:,:]        
        self.IRK_times = tmp[q**2+q:]

        self.lb = lb # borne inférieure de la grille
        self.ub = ub # borne supérieure de la grille
        self.dt = dt # pas de temps

        # On définit les paramètres du problème
        self.nu = nu # coefficient de diffusion
        self.q = q # ordre de la méthode Runge-Kutta

        # On définit les données d'entrainement
        self.x0 = x0 # position initiale
        self.u0 = u0 # solution initiale
        self.x1 = x1 # position secondaire
        self.u1 = u1 # solution secondaire

        # Transformation des données d'entraineement en tenseurs
        self.x0_tf = tf.convert_to_tensor(self.x0, dtype=tf.float32)
        self.u0_tf = tf.convert_to_tensor(self.u0, dtype=tf.float32)
        self.x1_tf = tf.convert_to_tensor(self.x1, dtype=tf.float32)
        self.u1_tf = tf.convert_to_tensor(self.u1, dtype=tf.float32)

        # Dummy variables pour les gradients avec des tenseurs de shape [None, self.q]
        self.dummy_x0_tf = tf.convert_to_tensor(np.zeros((256, self.q)), dtype=tf.float32)
        self.dummy_x1_tf = tf.convert_to_tensor(np.zeros((256, self.q)), dtype=tf.float32)

        # On définit les paramètres du réseau de neurones
        self.layers = layers # nombre de neurones par couche

        # On définit les paramètres de l'entraînement
        self.learning_rate = LEARNING_RATE # taux d'apprentissage
        self.num_epochs = NUM_EPOCHS # nombre d'époques
        self.batch_size = BATCH_SIZE # taille des batchs

        # On construit le réseau de neurones
        self.neuralNet = self.make_NN(layers)

        # TODO: COMMENTER
        self.U0_pred = self.net_U0(self.x0_tf) # N0 x q
        self.U1_pred = self.net_U1(self.x1_tf) # N1 x q

    def fwd_gradients_0(self, U, x):
        '''
        Méthode fwd_gradients_0
        -----------------------
        Cette méthode calcule les gradients de U par rapport à x.
        '''
        # g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        # return tf.gradients(g, self.dummy_x0_tf)[0]
        # meme chose que avec tf.GradientTape
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(U)
            tape.watch(self.dummy_x0_tf)
            U = self.neuralNet(x)
            g = tape.gradient(U, x, output_gradients=self.dummy_x0_tf)
            tape.watch(g)
            tape.watch(self.dummy_x0_tf)
            gg = tape.gradient(g, self.dummy_x0_tf)
        return gg

    def fwd_gradients_1(self, U, x):
        '''
        Méthode fwd_gradients_1
        -----------------------
        Cette méthode calcule les gradients de U par rapport à x.
        '''
        # g = tf.gradients(U, x, grad_ys=self.dummy_x1_tf)[0]
        # return tf.gradients(g, self.dummy_x1_tf)[0]
        # meme chose que avec tf.GradientTape
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(U)
            tape.watch(self.dummy_x1_tf)
            U = self.neuralNet(x)
            g = tape.gradient(U, x, output_gradients=self.dummy_x1_tf)
            tape.watch(g)
            tape.watch(self.dummy_x1_tf)
            gg = tape.gradient(g, self.dummy_x1_tf)
        return gg

    def net_U0(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        U = self.neuralNet(x)
        U_x = self.fwd_gradients_0(U, x)
        U_xx = self.fwd_gradients_0(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U0 = U - self.dt * tf.matmul(F, self.IRK_alpha.T)
        return U0

    def net_U1(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        U = self.neuralNet(x)
        U_x = self.fwd_gradients_1(U, x)
        U_xx = self.fwd_gradients_1(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U1 = U - self.dt * tf.matmul(F, (self.IRK_beta - self.IRK_alpha).T)
        return U1

    def make_NN(self, layers):
        '''
        Méthode make_NN
        ---------------
        Cette méthode construit le réseau de neurones.
        '''
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(layers[1], input_dim=layers[0], activation='tanh', kernel_initializer='glorot_normal'))
        for i in range(2, len(layers)):#TODO: CHECK IF NOT LEN(LAYERS)-1
            model.add(tf.keras.layers.Dense(layers[i], activation='tanh', kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1], activation='linear', kernel_initializer='glorot_normal'))

        return model

    def loss_func(self, a, b):
        self.U0_pred = self.net_U0(self.x0_tf)
        self.U1_pred = self.net_U1(self.x1_tf)
        return tf.reduce_sum(tf.square(self.u0_tf - self.U0_pred)) + \
                tf.reduce_sum(tf.square(self.u1_tf - self.U1_pred))

    def predict(self, X_star):
        '''
        Méthode predict
        ---------------
        Cette méthode permet de prédire la solution du problème à partir du
        réseau de neurones entraîné.
        '''
        u0_pred = self.net_U0(X_star[:,0:1])
        u1_pred = self.net_U1(X_star[:,0:1])
        return u0_pred, u1_pred

    def train(self, epochs, batch_size=1, learning_rate=1e-3, patience=10, verbose=True):
        '''
        Méthode train
        -------------
        Cette méthode permet d'entraîner le réseau de neurones.
        Avec patience = 10, on arrête l'entraînement si la fonction de coût
        n'a pas diminué pendant 10 époques.
        '''
        # On définit l'optimiseur
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        
        self.neuralNet.compile(optimizer=optimizer, loss=self.loss_func)

        self.neuralNet.fit(
            x=self.x1_tf,
            y=self.u1_tf,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=verbose
        )





################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    ############################################################################
    # PARAMETRES
    ############################################################################
    nu = 0.01/np.pi # coefficient de diffusion

    skip = 80 # Saut entre u0 et u1

    N0 = 256
    N1 = 256

    idx_t = 10

    ############################################################################
    # DATA INITIALIZATION
    ############################################################################
    # Importation des données
    data = sio.loadmat(DATA_PATH)

    # Affichage des données
    print("\nData : " + '-'*80)
    ## List of variables
    print("List of variables: ", data.keys())
    ## Shape of variables
    print("Shape of x: ", data['x'].shape)
    print("Shape of t: ", data['t'].shape)
    print("Shape of usol: ", data['usol'].shape)
    print('')

    # Extraction des données
    x_star = data['x'].flatten()[:,None] # x est un vecteur de 256 éléments. Donc unidimensionnel.
    t_star = data['t'].flatten()[:,None] # t est un vecteur de 100 éléments. Donc unidimensionnel.
    usol = np.real(data['usol']) # usol est une matrice de 256x100 éléments.
    ## usol est la solution exacte de l'équation de Burgers. Donc ground truth
    ## des données fournies par le fichier .mat (x et t).
    ## Les lignes correspondent à x et les colonnes à t.

    ############################################################################
    # DATA PROCESSING
    ############################################################################
    noise = 0.0 # Bruit ajouté aux données

    idx_x = np.random.choice(usol.shape[0], N0, replace=False) # Choix aléatoire de N0 éléments dans les lignes de usol
    x0 = x_star[idx_x,:] # x0 est un vecteur de N0 éléments. Donc unidimensionnel.
    u0 = usol[idx_x,idx_t][:,None] # u0 est un vecteur de N0 éléments. Donc unidimensionnel.
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1]) # Bruit ajouté aux données
    ## u0 est la solution exacte de l'équation de Burgers à t = t_star[idx_t]
    ## pour les N0 éléments choisis aléatoirement dans x_star.

    idx_x = np.random.choice(usol.shape[0], N1, replace=False) # Choix aléatoire de N1 éléments dans les lignes de usol
    x1 = x_star[idx_x,:] # x1 est un vecteur de N1 éléments. Donc unidimensionnel.
    u1 = usol[idx_x,idx_t+skip][:,None] # u1 est un vecteur de N1 éléments. Donc unidimensionnel.
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1]) # Bruit ajouté aux données
    ## u1 est la solution exacte de l'équation de Burgers à t = t_star[idx_t+skip]
    ## pour les N1 éléments choisis aléatoirement dans x_star.

    # Calcul de q
    dt = (t_star[idx_t+skip] - t_star[idx_t]).item() # dt est un scalaire. On a remplacé np.asscalar par .item().
    q = int(np.ceil(0.5 * np.log(np.finfo(float).eps) / np.log(dt))) # q est un scalaire
    print('q = ', q)

    # Définition de la structure du réseau de neurones
    layers = [1, 100, 100, 100, 100, q] # Nombre de neurones par couche

    # Affichage des données
    if VERIF_PLOTS:
        print("Data processing : " + '-'*80)
        ## Shape of variables
        print("Shape of x0: ", x0.shape)
        print("Shape of u0: ", u0.shape)
        print("Shape of x1: ", x1.shape)
        print("Shape of u1: ", u1.shape)
        print('')
        ## Affichage des données
        plt.figure(2)
        plt.plot(x0, u0, 'b.', label = 'Data (t = %.2f)' % (t_star[idx_t]))
        plt.plot(x1, u1, 'r.', label = 'Data (t = %.2f)' % (t_star[idx_t+skip]))
        plt.plot(x_star, usol[:,idx_t], 'b-', label = 'Exact (t = %.2f)' % (t_star[idx_t]))
        plt.plot(x_star, usol[:,idx_t+skip], 'r-', label = 'Exact (t = %.2f)' % (t_star[idx_t+skip]))
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.show()

    # Limites du domaine
    lb = x_star.min(0) # lb est un vecteur de 2 éléments qui désigne les limites inférieures du domaine.
    ub = x_star.max(0) # ub est un vecteur de 2 éléments qui désigne les limites supérieures du domaine.

    # Construction du modèle
    model = PINN(x0, u0, x1, u1, layers, dt, lb, ub, nu, q)

    # Entraînement du modèle
    model.train(10000)

    # Affichage des résultats
    if PLOTS:
        # Calculs
        u_pred, f_pred = model.predict(tf.convert_to_tensor(x_star, dtype=tf.float32))
        # Affichage
        
        plt.figure(2)
        plt.plot(x_star, usol[:,idx_t], 'b-', label = 'Exact (t = %.2f)' % (t_star[idx_t]))
        plt.plot(x_star, usol[:,idx_t+skip], 'r-', label = 'Exact (t = %.2f)' % (t_star[idx_t+skip]))

        plt.plot(x0, u0, 'b.', label = 'Data (t = %.2f)' % (t_star[idx_t]))
        plt.plot(x1, u1, 'r.', label = 'Data (t = %.2f)' % (t_star[idx_t+skip]))

        plt.plot(x_star, u_pred[:,0], 'k--', label = 'Prediction (t = %.2f)' % (t_star[idx_t+skip]))

        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.show()