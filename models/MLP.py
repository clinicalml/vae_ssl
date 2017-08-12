from AbstractMLP import * 

class MLP(AbstractMLP):

    def build_classifier(self, XL, Y):
        _, logbeta = self.build_inference_alpha(XL)
        probs = T.nnet.softmax(logbeta)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        loss= T.nnet.categorical_crossentropy(probs,Y) 
        accuracy = T.eq(T.argmax(probs,axis=1),Y)
        return probs, loss, accuracy

    def build_inference_alpha(self, X): 
        """
        return h(x), logbeta(h(x))
        """
        if not self._evaluating:
            X = self.dropout(X,self.params['input_dropout'])
            self._p(('Inference with dropout :%.4f')%(self.params['input_dropout']))


        with self.namespaces('h(x)'):
            hx = self.build_hidden_layers(X,diminput=self.params['dim_observations']
                                          ,dimoutput=self.params['q_dim_hidden']
                                          ,nlayers=self.params['q_layers'])

        with self.namespaces('h_logbeta'):
            h_logbeta = self.build_hidden_layers(hx,diminput=self.params['q_dim_hidden']
                                                  ,dimoutput=self.params['q_dim_hidden']
                                                  ,nlayers=self.params['alpha_inference_layers'])

        if not self._evaluating:
            h_logbeta = self.dropout(h_logbeta,self.params['dropout_logbeta']) 

        with self.namespaces('logbeta'):
            logbeta = self.linear(h_logbeta,diminput=self.params['q_dim_hidden']
                                            ,dimoutput=self.params['nclasses'])

        #clip to avoid nans
        logbeta = T.clip(logbeta,-5,5)

        self.tOutputs['logbeta'] = logbeta
        return hx, logbeta

