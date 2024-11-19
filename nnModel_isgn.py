class ISGN(nn.Module):
    def __init__(self, network_parameters, device):
        super(ISGN, self).__init__()
        self.device = device
        self.num_graph_iteration = network_parameters.graph_iteration
        self.num_sequence_iteration = network_parameters.sequence_iteration
        self.is_graph = True
        self.is_baseline = network_parameters.is_baseline
        if hasattr(network_parameters, 'is_test_version') and network_parameters.is_test_version:
            self.is_test_version = True
        else:
            self.is_test_version = False

        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_layers = network_parameters.note.layer
        self.note_hidden_size = network_parameters.note.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size
        self.final_hidden_size = network_parameters.final.size
        self.final_input = network_parameters.final.input
        self.encoder_size = network_parameters.encoder.size
        self.encoded_vector_size = network_parameters.encoded_vector_size
        self.encoder_input_size = network_parameters.encoder.input
        self.encoder_layer_num = network_parameters.encoder.layer
        self.time_regressive_size = network_parameters.time_reg.size
        self.time_regressive_layer = network_parameters.time_reg.layer
        self.num_edge_types = network_parameters.num_edge_types
        self.final_graph_margin_size = network_parameters.final.margin

        if self.is_baseline:
            self.final_graph_input_size = self.final_input + self.encoder_size + 8 + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size + 8 # tempo info
        else:
            self.final_graph_input_size = self.final_input + self.encoder_size + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size

        self.num_attention_head = network_parameters.num_attention_head
        # self.num_attention_head = 4

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
        )

        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, self.device, secondary_size=self.note_hidden_size)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, self.device)

        self.measure_attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.measure_rnn = nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)

        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU(),
            nn.Linear(self.encoder_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_graph_encoder = GatedGraph(self.encoder_size, self.num_edge_types, self.device)
        self.performance_measure_attention = ContextAttention(self.encoder_size, self.num_attention_head)

        self.performance_encoder = nn.LSTM(self.encoder_size, self.encoder_size, num_layers=self.encoder_layer_num,
                                           batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.beat_tempo_contractor = nn.Sequential(
            nn.Linear(self.final_graph_input_size - self.time_regressive_size * 2, self.time_regressive_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size, self.encoder_size, num_layers=1, bidirectional=False)

        self.initial_result_fc = nn.Sequential(
            nn.Linear(self.final_input, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),

            nn.Linear(self.encoder_size, self.output_size),
            nn.ReLU()
        )

        self.final_graph = GatedGraph(self.final_graph_input_size, self.num_edge_types, self.device,
                                      self.output_size + self.final_graph_margin_size)
        if self.is_baseline:
            self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size, self.time_regressive_size,
                                     num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
            self.final_measure_attention = ContextAttention(self.output_size, 1)
            self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)

            self.fc = nn.Sequential(
                nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
                nn.Dropout(DROP_OUT),
                nn.ReLU(),
                nn.Linear(self.final_graph_margin_size, self.output_size),
            )
        # elif self.is_test_version:
        else:
            self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size + 8, self.time_regressive_size,
                                     num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
            self.final_beat_attention = ContextAttention(self.output_size, 1)
            self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)
            self.tempo_fc = nn.Linear(self.time_regressive_size * 2, 1)

            self.fc = nn.Sequential(
                nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
                nn.Dropout(DROP_OUT),
                nn.ReLU(),
                nn.Linear(self.final_graph_margin_size, self.output_size-1),
            )
        # else:
        #     self.tempo_rnn = nn.LSTM(self.time_regressive_size + 3 + 5, self.time_regressive_size, num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
        #     self.final_beat_attention = ContextAttention(self.final_graph_input_size - self.time_regressive_size * 2, 1)
        #     self.tempo_fc = nn.Linear(self.time_regressive_size * 2, 1)
        #     # self.fc = nn.Linear(self.final_input + self.encoder_size + self.output_size, self.output_size - 1)
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.final_graph_input_size + 1, self.encoder_size),
        #         nn.Dropout(DROP_OUT),
        #         nn.ReLU(),
        #         nn.Linear(self.encoder_size, self.output_size - 1),
        #     )

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, return_z=False):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        section_numbers = [x.section for x in note_locations]
        num_notes = x.size(1)

        note_out, measure_hidden_out = self.run_graph_network(x, edges, measure_numbers, start_index)
        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                # zero_mean = torch.zeros(self.encoded_vector_size)
                # one_std = torch.ones(self.encoded_vector_size)
                # perform_z = self.reparameterize(zero_mean, one_std).to(self.device)
                perform_z = torch.Tensor(numpy.random.normal(size=self.encoded_vector_size)).to(self.device)
            # if type(initial_z) is list:
            #     perform_z = self.reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
            elif not initial_z.is_cuda:
                perform_z = torch.Tensor(initial_z).to(self.device).view(1,-1)
            else:
                perform_z = initial_z.view(1,-1)
            perform_mu = 0
            perform_var = 0
        else:
            perform_concat = torch.cat((note_out, y), 2).view(-1, self.encoder_input_size)
            perform_style_contracted = self.performance_contractor(perform_concat).view(1, num_notes, -1)
            perform_style_graphed = self.performance_graph_encoder(perform_style_contracted, edges)
            performance_measure_nodes = self.make_higher_node(perform_style_graphed, self.performance_measure_attention, beat_numbers,
                                                  measure_numbers, start_index, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
            perform_style_vector = self.performance_final_attention(perform_style_encoded)

            # perform_style_reduced = perform_style_reduced.view(-1,self.encoder_input_size)
            # perform_style_node = self.sum_with_attention(perform_style_reduced, self.perform_attention)
            # perform_style_vector = perform_style_encoded[:, -1, :]  # need check
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        if return_z:
            total_perform_z = [perform_z]
            for i in range(10):
                temp_z = self.reparameterize(perform_mu, perform_var)
                total_perform_z.append(temp_z)
            total_perform_z = torch.stack(total_perform_z)
            mean_perform_z = torch.mean(total_perform_z, 0, True)

            return mean_perform_z

        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)

        initial_output = self.initial_result_fc(note_out)
        num_measures = measure_numbers[start_index+num_notes-1] - measure_numbers[start_index] + 1
        # perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        # perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
        # measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        # measure_perform_style_spanned = self.span_beat_to_note_num(measure_perform_style, measure_numbers, num_notes, start_index)

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.time_regressive_size * 2)).to(self.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(self.device)


        num_beats = beat_numbers[start_index + num_notes - 1] - beat_numbers[start_index] + 1
        qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
        tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_beats, 1))
        beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_beats, 1))
        beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, start_index, TEMPO_IDX)

        total_iterated_output = [initial_output]

        # for i in range(2):

        if self.is_baseline:
            tempo_vector = x[:, :, TEMPO_IDX:TEMPO_IDX + 5].view(1, -1, 5)
            tempo_info_in_note = torch.cat((qpm_primo, tempo_primo, tempo_vector), 2)

            out_with_result = torch.cat(
                (note_out, perform_z_batched, tempo_info_in_note, initial_beat_hidden, initial_output, initial_margin), 2)

            for i in range(self.num_sequence_iteration):
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
                initial_out = out_with_result[:, :, -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:,:, -self.final_graph_margin_size:]

                margin_in_measure = self.make_higher_node(changed_margin, self.final_margin_attention, measure_numbers,
                                                 measure_numbers, start_index, lower_is_note=True)
                out_in_measure = self.make_higher_node(initial_out, self.final_measure_attention, measure_numbers,
                                                 measure_numbers, start_index, lower_is_note=True)

                out_measure_cat = torch.cat((margin_in_measure, out_in_measure), 2)

                out_beat_rnn_result, _ = self.tempo_rnn(out_measure_cat)
                out_beat_spanned = self.span_beat_to_note_num(out_beat_rnn_result, measure_numbers, num_notes, start_index)
                out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                             out_beat_spanned,
                                             out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                            2)
                final_out = self.fc(out_with_result)
                out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                             final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
                # out = torch.cat((out, trill_out), 2)
                total_iterated_output.append(final_out)
        else:
            out_with_result = torch.cat(
                # (note_out, measure_perform_style_spanned, initial_beat_hidden, initial_output, initial_margin), 2)
                (note_out, perform_z_batched, initial_beat_hidden, initial_output, initial_margin), 2)

            for i in range(self.num_sequence_iteration):
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
                initial_out = out_with_result[:, :,
                              -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

                margin_in_beat = self.make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                          beat_numbers, start_index, lower_is_note=True)
                out_in_beat = self.make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                       beat_numbers, start_index, lower_is_note=True)
                out_beat_cat = torch.cat((out_in_beat, margin_in_beat, beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), 2)
                out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
                tempo_out = self.tempo_fc(out_beat_rnn_result)
                tempos_spanned = self.span_beat_to_note_num(tempo_out, beat_numbers, num_notes, start_index)
                out_beat_spanned = self.span_beat_to_note_num(out_beat_rnn_result, beat_numbers, num_notes, start_index)
                out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                             out_beat_spanned,
                                             out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                            2)
                other_out = self.fc(out_with_result)

                final_out = torch.cat((tempos_spanned, other_out), 2)
                out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                             final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
                total_iterated_output.append(final_out)

        return final_out, perform_mu, perform_var, total_iterated_output

    def run_graph_network(self, nodes, adjacency_matrix, measure_numbers, start_index):
        # 1. Run feed-forward network by note level
        num_notes = nodes.shape[1]
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(self.device)
        notes_and_measure_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden = self.graph_1st(notes_and_measure_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)

            measure_nodes = self.make_higher_node(notes_hidden_cat, self.measure_attention, measure_numbers, measure_numbers,
                                                  start_index, lower_is_note=True)
            measure_hidden, _ = self.measure_rnn(measure_nodes)
            measure_hidden_spanned = self.span_beat_to_note_num(measure_hidden, measure_numbers, num_notes, start_index)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second),-1)
        return final_out, measure_hidden

    def encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self.reparameterize(mu, var)
        return z, mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # def decode_with_net(self, z, decode_network):
    #     decode_network
    #     return

    def sum_with_attention(self, hidden, attention_net):
        attention = attention_net(hidden)
        attention = self.softmax(attention)
        upper_node = hidden * attention
        upper_node_sum = torch.sum(upper_node, dim=0)

        return upper_node_sum

    def make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.stack(higher_nodes).view(1, -1, lower_hidden_size)

        return higher_nodes

    def span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0,i,beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat

    def note_tempo_infos_to_beat(self, y, beat_numbers, start_index, index=None):
        beat_tempos = []
        num_notes = y.size(1)
        prev_beat = -1
        for i in range(num_notes):
            cur_beat = beat_numbers[start_index+i]
            if cur_beat > prev_beat:
                if index is None:
                    beat_tempos.append(y[0,i,:])
                if index == TEMPO_IDX:
                    beat_tempos.append(y[0,i,TEMPO_IDX:TEMPO_IDX+5])
                else:
                    beat_tempos.append(y[0,i,index])
                prev_beat = cur_beat
        num_beats = len(beat_tempos)
        beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
        return beat_tempos
