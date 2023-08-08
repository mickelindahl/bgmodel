/*
 *  ml_module.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef ML_MODULE_H
#define ML_MODULE_H

// Includes from sli:
#include "slifunction.h"
#include "slimodule.h"

// Put your stuff into your own namespace.
namespace mynest
{
  
/**
 * Class defining your model.
 * @note For each model, you must define one such class, with a unique name.
 */
class Ml_Module : public SLIModule
{
public:
  // Interface functions ------------------------------------------
  
  /**
   * @note The constructor registers the module with the dynamic loader. 
   *       Initialization proper is performed by the init() method.
   */
  Ml_Module();
  
  /**
   * @note The destructor does not do much in modules.
   */
  ~Ml_Module() override;

  /**
   * Initialize module.
   * @param SLIInterpreter* SLI interpreter
   */
  void init( SLIInterpreter* ) override;

  /**
   * Return the name of your module.
   */
  const std::string name() const override;

};
} // namespace mynest

#endif