# Vertex Dynamics Simulation of Cell Monolayers
> Vertex dynamics simulation for modelling epithelial tissue dynamics (package description)


Modules:
- `primitives`: implements vertices (w/ distance), edges, cells (w/ perimeter and area), and monolayer (with energy, boundary constraints, and other parameters)
- `simulation`: tools for  simulating cellular vertex dynamics. Iterative algorithm implementations, cell monolayer generators (vertices). (anything else to add?!)

## Install

`pip install vertex_simulation` (not yet implemented)

## Using autograd with Vertex, Graph and Monolayer

In examples below, it is assumed that you've imported `primitives` module. You can do so with 
```
from vertex_simulation.primitives import *
```

### Initializing and working with `Vertex` and `Graph` objects

`Graph` objects implement graphs and stores its vertices as a `Vertex` object, and its edges as a `torch.tensor`. Both `Vertex` and `Graph` provide interface methods for working with torch's autograd, and methods for calculating edge lengths (or vertex to vertex distance in the case `Vertex`).

To illustrate how to use autograd, let's use `Vertex` object. We can define `Vertex` object with its location, or set and modify it later. Location is stored as `Vertex.x` property, a `torch.tensor`. `Vertex(location)` accepts `torch.tensors`, list of lists that are convertible to tensors (with optional keyword arg-s for `torch.tensor()`), and numpy.ndarrays (uses `torch.from_numpy()`)
- `v.x` (i.e. location for `v=Vertex(location)`) is assumed to be Nx2 array with float dtype (or any 2D array), and sizes are __not__ checked when set using `self.x`.
- Computing gradients, and resetting them to zeros. Example below demonstrates computing $\partial y/\partial v_{i,j}$ for $y = \sum_i\sum_j v_{i,j}^2$ using `torch.autograd`.

```python
v = Vertex([[3.,-1.],[0.1,0.]],requires_grad=True,dtype=torch.float32)
# do some calculation with v.x
y = torch.sum(v.x**2)
# calculate grad-s
y.backward()
print('dy/dx_i after y.backward():\n',v.grad())
# set grad-s to zeros (useful when you don't want to accumulate grad-s)
v.zero_grad_()
print('dy/dx_i after zeroing grad-s:\n',v.grad())
```

#### Vertex usage example
<a name="ex1">Trapped particle in a 2D fluid</a>

Let's assume linear drag, where force exerted by a spring is proportional to the velocity of the particle (drag force is $F_d=-b\frac{dx(t')}{dt'}$)
$$F_s(t') = -\nabla U = b\frac{dx(t')}{dt'}$$
where $U$ is the potential energy of the spring (e.g. optical trap) $U=k\cdot |r|^2$, where $r$ is the vector from equilibrium point, $o$, pointing to the current location of the particle, $x$. After re-defining time ($t'$) as a relative "scaled" time $t=\frac{t'}{b}$ and taking gradient of potential energy w.r.t. $x$, we can re-write the equation of motion (also let's set $o$ as origin $[0,0]$, then $r=x$)
$$\frac{dx(t)}{dt}=-2k(x-o)= -2kx$$
Let's numerically solve this equation ( _refer to the code cell below_ ). If the scale of the step size is chosen well (e.g. in the code below for large `k` use smaller `Dt`), solution $x(t)$ should converge to the equilibrium point $o$.

One way to model this system is to use two vertices, one for constant equilibrium point $o$, and a second vertex for particle position $x(t)$. To track potential energy gradient w.r.t. $x(t)$ we'll set `requires_grad=True` for the moving vertex, `v1` in the code below (this flag enables `torch`'s autograd to backpropagate the gradients).

```python
o  = Vertex(torch.tensor([[0,0]],dtype=torch.float64)) # equilibrium point (where U(x) is minimum)
v1 = Vertex(torch.tensor([[-3,3]],requires_grad=True,dtype=torch.float64)) # particles location
r = o.dist(v1)
print(f'distance from equilibrium (r):{r.item():.4f}')
print(f'o requires_grad? :{o.requires_grad()}',
      f'\nv1 requires_grad?:{v1.requires_grad()}',
      f'\nr requires_grad? :{r.requires_grad}')
# note that for Vertex self.requires_grad() is a function
```

In order to calculate gradients w.r.t. $x$, we need to set up a function that maps $x$ to some scalar value. In our example, this function is the potential energy function $U(x)$ (`energy(r)` below). Once `energy` function is evalutated, we need to call `backward()` on the returned `torch.tensor` to calculate (analytic) gradient of potential energy function at $x=v1$ (i.e. $\nabla_x U|_{x=v1}$; `dEdx` in code below)

```python
# Define energy
k = 1.0
energy = lambda r: k*r**2
E = energy(r)
print(f'Energy=kr^2 :{E.item():.4f}')
# compute gradients
E.backward()
dEdx = v1.grad().data
dxdt = -dEdx
print(f't=0: dE/dx={dEdx.tolist()} --> dx/dt=-dE/dx={dxdt.tolist()}')
```

An important point to keep in mind when using iterative methods (e.g. gradient descent) shown in the code below, is to remember to reset gradients accumulator to zeros. For `Vertex` objects its done with `Vertex.zero_grad_()`, if the vertex has `requires_grad=True` flag, calling this method sets all gradients of a given vertex to zeros. Otherwise it does nothing, e.g. gradients w.r.t. $o$ are kept as `None`, and they are not calculated.

```python
# Numerical integration
Dt = .16 # time step size
positions = [v1.x.tolist()]
t = [0]
Energies = []
print('Integration (Euler\'s method):')
for n in range(25):
    v1.zero_grad_()
    E = energy(o.dist(v1))  # elastic energy, o.dist(v1) is distance from vertex "o"
    Energies.append(E.item())
    E.backward()   # compute gradients
    dxdt = -v1.grad().data# dx/dt=-dE/dx
    if n%5==0:
        print(f't={Dt*n:.2f}:r={o.dist(v1).item():4.3f}; E={E.item():.2g}; dx/dt={dxdt}')
    # Update vertex position
    with torch.no_grad():
        v1.x += dxdt*Dt
    positions.append(v1.x.tolist()); t.append(t[-1]+Dt)
Energies.append( energy(o.dist(v1)).item() )
```

Results of the numerical integration above-- the evolution of the system in relative time, are shown below. Keep in mind that, in this simulation time is scaled by drag coefficient $b$, and for more accurate dynamics we need to use smaller `Dt` (or more accurate method for numerical integration).

```python
# Display the results
positions = np.array(positions).squeeze() # convert to a np array
fig = plt.figure(figsize=plt.figaspect(0.25))

# Energy as a function of position and particle trajectory
ax = fig.add_subplot(1, 3, 1, projection='3d')
# Plot the Energy surface
Xmesh,Ymesh = np.meshgrid(np.arange(-4,4,.25),np.arange(-4,4,.25))
Zmesh = k*((Xmesh-o.x[0,0].numpy())**2+(Ymesh-o.x[0,1].numpy())**2) # potential energy surface
ax.plot_surface(Xmesh,Ymesh, Zmesh, alpha=0.15)
# Plot trajectory of the vertex E,x1,x2
ax.plot(positions[:,0],positions[:,1],Energies,'ro-',alpha=.3)
ax.plot(positions[:1,0],positions[:1,1],Energies[:1],'mo',ms=15,label='start')
ax.plot(positions[-1:,0],positions[-1:,1],Energies[-1:],'bo',ms=15,label='end')
ax.set_xlabel('positions $x_1$'); ax.set_ylabel('positions $x_2$'); ax.set_zlabel('$Energy$');
plt.legend();

# Energy as function of time
ax = fig.add_subplot(1, 3, 2);
ax.plot(t,Energies);
ax.set_xlabel('time'); ax.set_ylabel('Energy')

# Vertex position (components) as a function of time
ax = fig.add_subplot(1, 3, 3)
ax.plot(t,positions);
plt.legend(['$x_1$','$x_2$']);
ax.set_ylabel('Positions $x_i$'); ax.set_xlabel('time');

plt.show()
```

#### `Graph` usage example
<a name="ex2">Attracting particles in a 2D fluid</a>

Now, let's evolve in time a system decribed by a potential
$$U=k\sum_{\forall ij|j\neq i}|x_i-x_j|^2=k\sum_{\forall ij|j\neq i}l^2_{ij}$$
where every vertex $i$ is connected to all the other vertices $j$ with edges $ij$, and $x_i$ is the position of vertex $i$ on a 2D plane (vector). Force balance equation for this system, same as in [Example 1](#ex1) is 
$$b\frac{dx(t')}{dt'}=-\nabla U$$
$\nabla U$ is a function of distances between all possible pairs of vertices (edge lengths $l_{ij}$, scalars). The equation of motion for every vertex is (with $t=t'/b$)
$$\frac{dx_i}{dt}=-k\sum_{\forall ij|j\neq i}2(x_i-x_j)= 2k\sum_{\forall ij|j\neq i}(x_j-x_i)$$
This system can be described by a complete graph, `G` in the code below. In order to demonstrate how to work with this type of systems, let's create a complete graph with $N_v$ vertices.

```python
np.random.seed(42) # let's seed RNG for sanity and reproducibility
Nv = 10 # number of vertices
Xv = np.random.uniform(0,1,(Nv,2)) # initial vertex potions sampled from uniform distribution [0,1)
edges = [[i,j] for i in range(Nv) for j in range(i+1,Nv) if i!=j] # list of edges for complete graph
plot_graph(Xv,edges) # plot vertices and edges
```

Now, let's solve $x(t)$ with Euler's method. Note that in the code below, `Dt` must be smaller for large $N_v$ (e.g. about $0.01$ or less for $N_v=10$, and about $0.001$ for $N_v=100$). Try changing the parameters (one at a time) and observe what happens.

```python
# initialize a graph
G = Graph(vertices=Vertex(torch.from_numpy(Xv).clone(),requires_grad=True, dtype=torch.float64), edges=torch.tensor(edges) )
G.vertices.requires_grad_(True) # turn on `Vertex` gradients; check its status with G.vertices.requires_grad()
print('Number of vertices:',G.vertices.x.size(0),'\nNumber of edges:', G.edges.size(0),
      '\nRequires grad?:',G.vertices.requires_grad())

# Define energy function
k=1.0
energy = lambda l: k*torch.sum(l**2) # E = k sum(l_ij ^2)
# Numerical integration
Dt = 2**-10 # time step size
positions = [G.vertices.x.clone()]
t = [0]
Energies = []
print('Integration (Euler\'s method):')
for n in range(256):
    G.set_zero_grad_() # reset grad accumulator
    E = energy(G.length())  # total potential energy of the system
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -G.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        G.vertices.x += dxdt*Dt
    positions.append(G.vertices.x.clone())
    t.append(t[-1]+Dt)
    if n%32==0:
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():1.1g}; aver |dx/dt|= {mean_grad:1.1g}')
Energies.append( energy(G.length()).item() )
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
```

Results for numerical integration above as a movie of the graph $G$:

```python
HTML(f_anim.to_jshtml()) # using HTML from IPython.display and matplotlib's animation module
```

### `Monolayer` objects

`Monolayer` object stores vertices, edges, and cells and implements methods for working wiht torch's autograd. In order to demonstrate how to use `Monolayer` objects, we start with generating cells. Here we will use Voronoi tessellation.

```python
from scipy.spatial import Voronoi,voronoi_plot_2d

v_seeds=np.array([[np.sqrt(3)/2,5.5], [1.5*np.sqrt(3),5.5], [0.,4.],
                  [np.sqrt(3),4.],[2*np.sqrt(3),4.],[-np.sqrt(3)/2,2.5],
                  [np.sqrt(3)/2,2.5],[1.5*np.sqrt(3),2.5],[2.5*np.sqrt(3),2.5],
                  [0.,1.],[2*np.sqrt(3),1.], [np.sqrt(3),1.]])

vrn = Voronoi(v_seeds)
voronoi_plot_2d(vrn)
plt.show()
```

After obtaining the Voronoi tesselation, use `VoronoiRegions2Edges` to convert regions into a `Monolayer` (and `Graph`) compatible edges and cells representations:

```python
edge_list,cells = VoronoiRegions2Edges(vrn.regions) # convert regions to edges and cells
print(cells)
```

`cells` is the `dict` of edge indices. Negative edge indices indicate reversed vertex order:

```python
verts = Vertex(vrn.vertices)
edges = torch.tensor(edge_list)

plt.figure(figsize=[5,5])
plot_graph(verts.x,edges)

# vertex indices
for k,v in enumerate(vrn.vertices):
    plt.text(v[0]+.1,v[1]+.1,f"{k}",c='b',ha='center',alpha=.5)

# cell edges
for c in cells:
    cell_edges = edges[np.abs(cells[c])-1,:] # edge indices (without direction)
    if np.any(np.sign(cells[c])<0):
        # reverse vertex order for negative edges
        tmp = cell_edges[np.sign(cells[c])<0,:].clone()
        cell_edges[np.sign(cells[c])<0,0]=tmp[:,1]
        cell_edges[np.sign(cells[c])<0,1]=tmp[:,0]
    cell_xy = torch.mean(verts.x[cell_edges[:,0],:],0)
    for e in cell_edges:
        e_xy = torch.mean(verts.x[e,:],0)*.65+cell_xy*.35
        plt.text(e_xy[0],e_xy[1],f"{e.numpy()}",ha='center')
plt.show()
```

#### `Monolayer` dynamics example

```python
import networkx as nx
import matplotlib.animation as animation

np.random.seed(42)# let's seed RNG for sanity and reproducibility

v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2
# define cell monolayer
cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)
fig = plt.figure(figsize=[3,3])
nx.draw(Gnx,pos,node_size=10,node_color='#FF00FF',edge_color='#51C5FF')
```

### Passive forces case

```python
# let's seed RNG for sanity and reproducibility
np.random.seed(42)

# define cell monolayer
v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2

cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)

fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')
plt.show()
plt.close()

# Define energy function
energy = lambda p,a: torch.sum(.01*(p)**2)+torch.sum((a-2.3)**2) #
# Numerical integration
Dt = 2**-3 # time step size
t = [0]
Energies = []
Forces = []
verts_t =[]
verts_frames=[]
print('Integration (Euler\'s method):')
t_total = 2**8
cell_graph.vertices.requires_grad_(True)
for n in range(t_total):
    cell_graph.set_zero_grad_() # reset grad accumulator
    E = energy(cell_graph.perimeter(),cell_graph.area())  # total potential energy of the system
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -cell_graph.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        cell_graph.vertices.x += dxdt*Dt
        Forces.append(torch.norm(dxdt,dim=1).mean().item())
    if (n+1)%32==0 or n==0:
        verts_t.append(cell_graph.vertices.x.detach().clone())
        verts_frames.append(t[-1])
    if round((n+1)%(t_total/12))==0:
        #plt.figure(figsize=figsize)
        #plot_graph_as_quiver(init_state,quiver_kwargs=quiver_kwargs)
        #plot_graph_as_quiver(cell_graph)
        #plt.axis(axs_lims)
        #plt.show()
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():5.4g}; aver |dx/dt|= {mean_grad:3.2g}')
    t.append(t[-1]+Dt)

Energies.append( energy(cell_graph.perimeter(),cell_graph.area()).item() )
plt.figure(figsize=[5,3])
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
# add forces (except last frame)
ax2=plt.gca().twinx()
ax2.set_ylabel('Average $|F|$',color='red')
ax2.plot(t[:-1],Forces,'r-',alpha=.4,lw=4)
plt.show()
# Print final Perimeters and Areas
print(f"Perimeters:{cell_graph.perimeter().detach().squeeze()}\nAreas:{cell_graph.area().detach()}")
```

```python
def draw(i):
    pos = dict(zip(range(verts_t[i].shape[0]),verts_t[i].numpy()))
    ax.cla()
    ax.axis('off')
    ax.set_title(f'Epoch:{verts_frames[i]:2.3f}')
    nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')

# Networkx's edge ordering is different
edge_idx = dict(zip([tuple(e) for e in cell_graph.edges.tolist()],range(cell_graph.edges.shape[0])))
edge_idx_order = [edge_idx[e if e in edge_idx else (e[1],e[0])] for e in Gnx.edges ]
def draw_w_tension(i):
    pos = dict(zip(range(verts_t[i].shape[0]),verts_t[i].numpy()))
    ax.cla()
    ax.axis('off')
    ax.set_title(f'Epoch:{verts_frames[i]:2.3f}')
    #node_color=range(24), node_size=800, cmap=plt.cm.Blues
    nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',
            edge_color=line_tensions[i].numpy()[edge_idx_order],edge_cmap=plt.cm.bwr)
```

```python
fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
draw(0)  # draw the prediction of the first epoch
plt.show()
plt.close()

ani_passive = animation.FuncAnimation(fig, draw, interval=200,
                              frames = range(0,len(verts_t),max(1,round(len(verts_t)/64))))
```

```python
print('Passive system (foam)')
HTML(ani_passive.to_jshtml()) # using HTML from IPython.display and matplotlib's animation module
```

### Active anisotropic forces

#### Edge direction is independent

```python
# let's seed RNG for sanity and reproducibility
np.random.seed(42)

# define cell monolayer
v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2

cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)

fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')
plt.show()
plt.close()

# Define energy function
omega = torch.tensor([np.pi/2],dtype=cell_graph.vertices.x.dtype)
def monolayer_energy(Perm,Area,Leng,Tau,direction):
    dir_coeff = torch.abs(direction.detach()[:,1])/torch.norm(direction.detach(),dim=1)
    gamma_ij_by_lij = (Leng*dir_coeff.view(-1,1))*torch.cos(omega*Tau)**2
    return (torch.sum(.01*(Perm)**2) + torch.sum((Area-2.3)**2) + torch.sum(gamma_ij_by_lij)), dir_coeff

# Numerical integration
Dt = 2**-3 # time step size
t = [0]
Energies = []
Forces = []
verts_t =[]
verts_frames=[]
line_tensions=[]
print('Integration (Euler\'s method):')
t_total = 2**8
cell_graph.vertices.requires_grad_(True)
for n in range(t_total):
    cell_graph.set_zero_grad_() # reset grad accumulator
    # total potential energy of the system:
    E,edge_tensions = monolayer_energy(cell_graph.perimeter(),cell_graph.area(),
                         cell_graph.length(),t[-1],
                        cell_graph.direction()) 
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -cell_graph.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        cell_graph.vertices.x += dxdt*Dt
        Forces.append(torch.norm(dxdt,dim=1).mean().item())
    
    if (n+1)%32==0:
        verts_t.append(cell_graph.vertices.x.detach().clone())
        verts_frames.append(t[-1])
        line_tensions.append(edge_tensions.detach().clone())
    if round((n+1)%(t_total/10))==0:
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():5.4g}; aver |dx/dt|= {mean_grad:3.2g}')
    t.append(t[-1]+Dt) # update last frame time

Energies.append( monolayer_energy( cell_graph.perimeter(), cell_graph.area(),
                                  cell_graph.length(),t[-1],cell_graph.direction())[0].item() )
plt.figure(figsize=[5,3])
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
# add forces (except last frame)
ax2=plt.gca().twinx()
ax2.set_ylabel('Average $|F|$',color='red')
ax2.plot(t[:-1],Forces,'r-',alpha=.4,lw=2)
plt.show()
# Print final Perimeters and Areas
print(f"Perimeters:{cell_graph.perimeter().detach().squeeze()}\nAreas:{cell_graph.area().detach()}")
```

```python
ani_dir_no_grad = animation.FuncAnimation(fig, draw_w_tension, interval=200,
                              frames = range(0,len(verts_t),max(1,round(len(verts_t)/64))))
```

```python
print('Direction is independent of positions (gradient==0)')
HTML(ani_dir_no_grad.to_jshtml()) # using HTML from IPython.display and matplotlib's animation module
```

#### with anisotropy and differentiable edge direction

```python
# let's seed RNG for sanity and reproducibility
np.random.seed(42)

# define cell monolayer
v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2

cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)

fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')
plt.show()
plt.close()

# Define energy function
omega = torch.tensor([np.pi/2],dtype=cell_graph.vertices.x.dtype)
def monolayer_energy(Perm,Area,Leng,Tau,direction):
    dir_coeff = torch.abs(direction[:,1])/torch.norm(direction,dim=1)
    gamma_ij_by_lij = (Leng*dir_coeff.view(-1,1))*torch.cos(omega*Tau)**2
    return (torch.sum(.01*(Perm)**2) + torch.sum((Area-2.3)**2) + torch.sum(gamma_ij_by_lij)), dir_coeff

# Numerical integration
Dt = 2**-3 # time step size
t = [0]
Energies = []
Forces = []
verts_t =[]
verts_frames=[]
line_tensions=[]
print('Integration (Euler\'s method):')
t_total = 2**8

cell_graph.vertices.requires_grad_(True)
for n in range(t_total):
    cell_graph.set_zero_grad_() # reset grad accumulator
    # total potential energy of the system:
    E,dir_coeff = monolayer_energy(cell_graph.perimeter(),cell_graph.area(),
                         cell_graph.length(),t[-1],
                        cell_graph.direction()) 
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -cell_graph.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        cell_graph.vertices.x += dxdt*Dt
        Forces.append(torch.norm(dxdt,dim=1).mean().item())
    
    if (n+1)%32==0:
        verts_t.append(cell_graph.vertices.x.detach().clone())
        verts_frames.append(t[-1])
        line_tensions.append(dir_coeff.detach().clone())
    if round((n+1)%(t_total/8))==0:
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():5.4g}; aver |dx/dt|= {mean_grad:3.2g}')
    t.append(t[-1]+Dt) # update last frame time

Energies.append( monolayer_energy( cell_graph.perimeter(), cell_graph.area(),
                                  cell_graph.length(),t[-1],cell_graph.direction())[0].item() )
plt.figure(figsize=[5,3])
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
# add forces (except last frame)
ax2=plt.gca().twinx()
ax2.set_ylabel('Average $|F|$',color='red')
ax2.plot(t[:-1],Forces,'r-',alpha=.4,lw=2)
plt.show()
# Print final Perimeters and Areas
print(f"Perimeters:{cell_graph.perimeter().detach().squeeze()}\nAreas:{cell_graph.area().detach()}")
```

```python
ani_dir_grad = animation.FuncAnimation(fig, draw_w_tension, interval=200,
                              frames = range(0,len(verts_t),max(1,round(len(verts_t)/64))))
```

```python
print('Direction has gradient w.r.t. positions')
HTML(ani_dir_grad.to_jshtml()) # using HTML from IPython.display and matplotlib's animation module
```

#### Anisotropic active forces with lower amplitude

```python
# let's seed RNG for sanity and reproducibility
np.random.seed(42)

# define cell monolayer
v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2

cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)

fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')
plt.show()
plt.close()

# Define energy function
omega = torch.tensor([np.pi/2],dtype=cell_graph.vertices.x.dtype)
def monolayer_energy(Perm,Area,Leng,Tau,direction):
    dir_coeff = torch.abs(direction[:,1])/torch.norm(direction,dim=1)/5
    gamma_ij_by_lij = (Leng*dir_coeff.view(-1,1))*torch.cos(omega*Tau)**2
    return (torch.sum(.01*(Perm)**2) + torch.sum((Area-2.3)**2) + torch.sum(gamma_ij_by_lij)), dir_coeff

# Numerical integration
Dt = 2**-3 # time step size
t = [0]
Energies = []
Forces = []
verts_t =[]
verts_frames=[]
line_tensions=[]
print('Integration (Euler\'s method):')
t_total = 2**8

cell_graph.vertices.requires_grad_(True)
for n in range(t_total):
    cell_graph.set_zero_grad_() # reset grad accumulator
    t.append(t[-1]+Dt) # update last frame time
    # total potential energy of the system:
    E,dir_coeffs = monolayer_energy(cell_graph.perimeter(),cell_graph.area(),
                         cell_graph.length(),t[-1],
                        cell_graph.direction()) 
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -cell_graph.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        cell_graph.vertices.x += dxdt*Dt
        Forces.append(torch.norm(dxdt,dim=1).mean().item())
    
    if (n+1)%32==0:
        verts_t.append(cell_graph.vertices.x.detach().clone())
        verts_frames.append(t[-1])
        line_tensions.append(dir_coeffs.detach().clone())
    if round((n+1)%(t_total/8))==0:
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():5.4g}; aver |dx/dt|= {mean_grad:3.2g}')
Energies.append( monolayer_energy( cell_graph.perimeter(), cell_graph.area(),
                                  cell_graph.length(),t[-1],cell_graph.direction())[0].item() )
plt.figure(figsize=[5,3])
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
# add forces (except last frame)
ax2=plt.gca().twinx()
ax2.set_ylabel('Average $|F|$',color='red')
ax2.plot(t[:-1],Forces,'r-',alpha=.4,lw=2)
plt.show()
# Print final Perimeters and Areas
print(f"Perimeters:{cell_graph.perimeter().detach().squeeze()}\nAreas:{cell_graph.area().detach()}")
```

```python
ani_dir_grad_low_amplitude = animation.FuncAnimation(fig, draw_w_tension, interval=200,
                              frames = range(0,len(verts_t),max(1,round(len(verts_t)/64))))
```

```python
print('Direction has gradient w.r.t. positions (low line tension: 1/5)')
# using HTML from IPython.display and matplotlib's animation module
HTML(ani_dir_grad_low_amplitude.to_jshtml()) 
```

#### Random edges with active tension

```python
e_ij_coeff = torch.zeros(len(edge_list),1,dtype=cell_graph.vertices.x.dtype)
e_ij_on = np.random.rand(len(edge_list),1)<.2
e_ij_coeff[e_ij_on] = 1.0

```

```python
# define cell monolayer
v_x,regions =unit_hexagons(4,4) # 4x4 hexagons
# convert Voronoi regions to cells and edges
edge_list,cells = VoronoiRegions2Edges(regions)
# perturb vertices
np.random.seed(42) # let's seed RNG for sanity and reproducibility
v_x += np.random.randn(v_x.shape[0], v_x.shape[1])*.2

cell_graph = Monolayer(vertices=Vertex(v_x.copy()), edges=torch.tensor(edge_list), cells=cells)

Gnx,pos=graph2networkx_with_pos(cell_graph)

fig = plt.figure(figsize=[5,5])
fig.clf()
ax = fig.subplots()
ax.axis(False);
nx.draw(Gnx,pos,node_size=10,ax=ax,node_color='#FF00FF',edge_color='#51C5FF')
plt.show()
plt.close()

# Define energy function
omega = torch.tensor([np.pi/2],dtype=cell_graph.vertices.x.dtype)
e_ij_coeff = torch.zeros(len(edge_list),1,dtype=cell_graph.vertices.x.dtype)
e_ij_on = np.random.rand(len(edge_list),1)<.2
e_ij_coeff[e_ij_on] = 1.0
e_ij_phase = torch.zeros_like(e_ij_coeff)
e_ij_phase[e_ij_on] = torch.rand((e_ij_on.sum(),),dtype=cell_graph.vertices.x.dtype)
def monolayer_energy(Perm,Area,Leng,Tau,direction):
    gamma_ij_by_lij = (Leng*e_ij_coeff/5)*torch.cos(omega*Tau+e_ij_phase)**2
    return (torch.sum(.1*(Perm)**2) + torch.sum((Area-2.3)**2) + torch.sum(gamma_ij_by_lij)),gamma_ij_by_lij.squeeze()

# Numerical integration
Dt = 2**-5 # time step size
t = [0]
Energies = []
Forces = []
verts_t =[]
verts_frames=[]
line_tensions=[]
print('Integration (Euler\'s method):')
t_total = 2**10

cell_graph.vertices.requires_grad_(True)
for n in range(t_total):
    cell_graph.set_zero_grad_() # reset grad accumulator
    t.append(t[-1]+Dt) # update last frame time
    # total potential energy of the system:
    E,dir_coeffs = monolayer_energy(cell_graph.perimeter(),cell_graph.area(),
                         cell_graph.length(),t[-1],
                        cell_graph.direction()) 
    Energies.append(E.item()) # E(t-1)
    E.backward() # compute gradients
    dxdt = -cell_graph.get_vertex_grad() # dx/dt=-dE/dx
    # Update vertex position
    with torch.no_grad():
        cell_graph.vertices.x += dxdt*Dt
        Forces.append(torch.norm(dxdt,dim=1).mean().item())
    
    if (n+1)%32==0:
        verts_t.append(cell_graph.vertices.x.detach().clone())
        verts_frames.append(t[-1])
        line_tensions.append(dir_coeffs.detach().clone())
    if round((n+1)%(t_total/8))==0:
        mean_grad = torch.norm(dxdt,dim=1).mean().item()
        print(f't={t[-1]:2.3f}: E={E.item():5.4g}; aver |dx/dt|= {mean_grad:3.2g}')
Energies.append( monolayer_energy( cell_graph.perimeter(), cell_graph.area(),
                                  cell_graph.length(),t[-1],cell_graph.direction())[0].item() )
plt.figure(figsize=[5,3])
plt.plot(t,Energies);plt.xlabel('time');plt.ylabel('energy');
# add forces (except last frame)
ax2=plt.gca().twinx()
ax2.set_ylabel('Average $|F|$',color='red')
ax2.plot(t[:-1],Forces,'r-',alpha=.4,lw=2)
plt.show()
# Print final Perimeters and Areas
print(f"Perimeters:{cell_graph.perimeter().detach().squeeze()}\nAreas:{cell_graph.area().detach()}")
```

```python
ani_dir_grad_low_mem_low = animation.FuncAnimation(fig, draw_w_tension, interval=200,
                              frames = range(0,len(verts_t),max(1,round(len(verts_t)/128))))
```

```python
print('Random edges with active tension')
# using HTML from IPython.display and matplotlib's animation module
HTML(ani_dir_grad_low_mem_low.to_jshtml()) 
```
