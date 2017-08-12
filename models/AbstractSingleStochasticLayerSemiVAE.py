from AbstractSemiVAE import * 

class AbstractSingleStochasticLayerSemiVAE(AbstractSemiVAE):

    def build_generative(self, alpha, Z):
        """
        Build subgraph to estimate conditional params
        """
        with self.namespaces('p(x|z,alpha)'):

            # first transform Z into another embedding layer
            with self.namespaces('h(z)'):        
                Z = self.build_hidden_layers(Z,diminput=self.params['dim_stochastic']
                                             ,dimoutput=self.params['p_dim_hidden']
                                             ,nlayers=self.params['z_generative_layers']
                                             ,normalization=self.params['p_normlayers'])

            # combine alpha and Z
            h = T.concatenate([alpha,Z],axis=1)

            # hidden layers for p(x|z,alpha)
            with self.namespaces('h(h(z),alpha)'):        
                h = self.build_hidden_layers(h,diminput=self.params['dim_stochastic']+self.params['nclasses']
                                             ,dimoutput=self.params['p_dim_hidden']
                                             ,nlayers=self.params['p_layers']
                                             ,normalization=self.params['p_normlayers'])

            # calculate emission probability parameters
            if self.params['data_type']=='real':
                with self.namespaces('gaussian'):        
                    with self.namespaces('hidden'):
                        mu = self.linear(h,diminput=self.params['p_dim_hidden']
                                           ,dimoutput=self.params['dim_observations'])
                        logcov2 = self.linear(h,diminput=self.params['p_dim_hidden']
                                                ,dimoutput=self.params['dim_observations'])
                    params = {'mu':mu,'logcov2':logcov2}
            else:
                with self.namespaces('bernoulli'):        
                    with self.namespaces('hidden'):
                        h = self.linear(h,diminput=self.params['p_dim_hidden']
                                          ,dimoutput=self.params['dim_observations'])
                        p = T.nnet.sigmoid(h)
                    params = {'p':p}
            return params


    def build_inference_Z(self,alpha,hx):
        """
        return q(z|alpha,h(x))
        """

        if not self._evaluating:
            hx = self.dropout(hx,self.params['dropout_hx'])

        with self.namespaces('hz(hx)'):
            hz = self.build_hidden_layers(hx,diminput=self.params['q_dim_hidden']
                                           ,dimoutput=self.params['q_dim_hidden']
                                           ,nlayers=self.params['hz_inference_layers'])

        with self.namespaces('hz(alpha)'):
            alpha_embed = self.linear(alpha,diminput=self.params['nclasses']
                                            ,dimoutput=self.params['q_dim_hidden'])
            
        # concatenate hz and alpha_embed
        hz_alpha = T.concatenate([alpha_embed,hz],axis=1) 

        # infer mu and logcov2 for q(z|alpha,x)
        with self.namespaces("q(z|alpha,x)"):
            q_Z_h = self.build_hidden_layers(hz_alpha,diminput=2*self.params['q_dim_hidden']
                                                    ,dimoutput=self.params['q_dim_hidden']
                                                    ,nlayers=self.params['z_inference_layers'])

            diminput = self.params['q_dim_hidden']
            dimoutput = self.params['dim_stochastic']

            with self.namespaces('mu'):
                mu = self.linear(q_Z_h,diminput=self.params['q_dim_hidden']
                                       ,dimoutput=self.params['dim_stochastic'])
            with self.namespaces('logcov2'):
                logcov2 = self.linear(q_Z_h,diminput=self.params['q_dim_hidden']
                                              ,dimoutput=self.params['dim_stochastic'])

        return mu, logcov2


