import jax
from typing import NamedTuple, List

class WCNF(NamedTuple):     
  w: jax.Array
  indices: jax.Array

class Jsp(NamedTuple):
  data: jax.Array
  indices: jax.Array

class Jhdata(NamedTuple):
  J: List[Jsp] 		       
  h: jax.Array 			 		 
  Jat: List[Jsp]    	

class Sekey(NamedTuple):
	s: jax.Array
	e: jax.Array
	key: jax.Array

class Seminkey(NamedTuple):
	s: jax.Array
	min_s: jax.Array
	e: jax.Array
	min_e: jax.Array
	key: jax.Array
  
class Seminokey(NamedTuple):
	s: jax.Array
	min_s: jax.Array
	e: jax.Array
	min_e: jax.Array
