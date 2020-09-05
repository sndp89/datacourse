#!/usr/bin/env python3
# coding=utf-8

from IPython import display
import graphviz

graphs = {
    # Intro to TF notebook
    "add-op": """
        digraph g {
            x [shape=plaintext,label=""]
            y [shape=plaintext,label=""]
            z [shape=plaintext,label=""]
            x -> "tf.add" -> z;
            y -> "tf.add";
            rankdir=LR
        }""",
    "const-op": """
        digraph g {
            z [shape=plaintext,label=""]
            "tf.constant" -> z;
            rankdir=LR
        }""",
    "simple-graph": """
        digraph g {
            x -> add;
            y -> add;
            sum_ [shape=plaintext,label=""];
            add -> sum_ [label="sum_"];
            rankdir=LR
        }""",
    "simple-graph-run": """
        digraph g {
            x -> add [label="2"];
            y -> add [label="3"];
            5 [shape=plaintext];
            add -> 5 ;
            rankdir=LR
        }""",
    "exercise": """
        digraph g {
            x -> add;
            y -> add;
            add -> multiply;
            y -> multiply;
            out [shape=plaintext,label=""];
            multiply -> out;
            rankdir=LR
        }""",
    # RNN notebook
    "feed-forward": """
        digraph g {
            node [style=filled, label="", shape=circle]
            a [fillcolor=gray, shape=square];
            b [fillcolor="#91bfdb"];
            b1 [fillcolor="#91bfdb"];
            c [fillcolor="#fc8d59"];
            a -> b -> b1 -> c;
            rankdir=LR
        }""",
    "recurrent": """
        digraph g {
            node [style=filled, label="", shape=circle]
            a [fillcolor=gray, shape=square];
            b [fillcolor="#91bfdb"];
            c [fillcolor="#fc8d59"];
            a -> b -> c;
            b -> b [dir=back];
            rankdir=LR
        }""",
    "unrolled": """
        digraph g {
            node [fillcolor=gray, style=filled, label="", shape=square]
            a1; a2; a3; a4;
            node [fillcolor="#91bfdb", shape=circle]
            { rank = same; b1; b2; b3; b4; }
            node [fillcolor="#fc8d59"]
            c1; c2; c3; c4;

            a1 -> b1 -> c1;
            a2 -> b2 -> c2;
            a3 -> b3 -> c3;
            a4 -> b4 -> c4;
            b1 -> b2;
            b2 -> b3
            b3 -> b4;
            rankdir=LR
        }""",
    "classification": """
        digraph g {
            node [fillcolor=gray, style=filled, label="", shape=square]
            a1; a2; a3; a4;
            node [fillcolor="#91bfdb", shape=circle]
            { rank = same; b1; b2; b3; b4; }
            node [fillcolor="#fc8d59"]
            c4;

            a1 -> b1;
            a2 -> b2;
            a3 -> b3;
            a4 -> b4 -> c4;
            b1 -> b2;
            b2 -> b3
            b3 -> b4;
            rankdir=LR
        }""",
    "generation": """
        digraph g {
            node [fillcolor=gray, style=filled, label="", shape=square]
            a1;
            node [fillcolor="#91bfdb", shape=circle]
            { rank = same; b1; b2; b3; b4; }
            node [fillcolor="#fc8d59"]
            c1; c2; c3; c4;

            a1 -> b1 -> c1;
            c1 -> b2 [style=dashed, headport=w];
            b2 -> c2;
            c2 -> b3 [style=dashed, headport=w];
            b3 -> c3;
            c3 -> b4 [style=dashed, headport=w];
            b4 -> c4;
            b1 -> b2;
            b2 -> b3
            b3 -> b4;
            rankdir=LR
        }""",
    "translation": """
        digraph g {
            node [fillcolor=gray, style=filled, label="", shape=square]
            a1; a2; a3[label="end", fillcolor=white];
            node [fillcolor="#91bfdb", shape=circle]
            { rank = same; b1; b2; b3; b4; }
            node [fillcolor="#fc8d59"]
            c3; c4;

            a1 -> b1;
            a2 -> b2;
            a3 -> b3 -> c3;
            c3 -> b4 [style=dashed, headport=w];
            b4 -> c4;
            b1 -> b2;
            b2 -> b3
            b3 -> b4;
            rankdir=LR
        }""",
    "LSTM": u"""
        digraph g {
            node [style=filled, shape=circle, label="", fillcolor="#91bfdb"]

            {
                node [shape=plaintext,label="",fillcolor=none]
                a1; c1;
            }

            subgraph cluster0 {
                //node [style=filled,color=white];
                color="#dddddd";
                style=filled;
                label="LSTM Cell"

                input [shape=point];

                {
                    node [label="∫", fillcolor=white]
                    input_nonlin;
                    output_nonlin;
                }

                input -> input_nonlin -> input_gate;
                input -> input_gate;

                input_gate -> state -> output_nonlin -> output_gate;
                state -> state [dir=back, label="state", headport=e, tailport=w];
                input-> output_gate;
            }

            edge [color="black:invis:black:invis:black", arrowsize=2]
            a1 -> input;
            output_gate -> c1;

            rankdir=LR
        }""",
    "LSTM_full": u"""
       digraph g {
            node [style=filled, shape=circle, label="", fillcolor="#91bfdb"]

            {
                node [shape=plaintext,label=<<i>x</i><sub><font point-size="12"><i>i</i></font></sub>>,fillcolor=none]
                x;
            }

            {
                node [shape=plaintext,label=<<i>h</i><sub><font point-size="12"><i>i</i>-1</font></sub>>,fillcolor=none]
                h_in;
            }

            {
                node [shape=plaintext,label=<<i>h</i><sub><font point-size="12"><i>i</i></font></sub>>,fillcolor=none]
                h_out;
            }

            subgraph cluster0 {
                //node [style=filled,color=white];
                color="#dddddd";
                style=filled;
                label="LSTM Cell"

                input [shape=point];

                {
                    node [fillcolor=white, shape=ellipse]
                    update_tanh [label="tanh"];
                    output_tanh [label="tanh"];
                    input_gate [label=<<i>g<sub><font point-size="12">input</font></sub></i>>];
                    output_gate [label=<<i>g<sub><font point-size="12">output</font></sub></i>>];
                    forget_gate [label=<<i>g<sub><font point-size="12">forget</font></sub></i>>];
                }

                {
                    node [label="ⓧ", fillcolor=white]
                    state_in_prod;
                    update_prod;
                    out_prod;
                }

                {
                    node [label="+", fillcolor=white]
                    plus;
                }

                {
                    node [shape=square,label=<<i>C</i><sub><font point-size="12"><i>i</i>-1</font></sub>>,fillcolor=none]
                    c_in;
                }

                {
                    node [shape=square,label=<<i>C</i><sub><font point-size="12"><i>i</i></font></sub>>,fillcolor=none]
                    c_out;
                }

                {rank=same; output_tanh; out_prod; }


                input -> forget_gate -> state_in_prod;
                state_in_prod -> plus;
                input -> input_gate -> update_prod -> plus;
                input -> update_tanh -> update_prod;
                input -> output_gate -> out_prod;
                plus -> output_tanh -> out_prod;

                edge [color="black:black:black", arrowsize=1.]

                {rank=same; forget_gate -> update_tanh -> input_gate [style=invis] }
                {rank=same; c_in -> state_in_prod}

                c_in -> c_out [style=invis]

                {rank=same; c_out -> plus [dir=back] }

            }

            edge [color="black:invis:black:invis:black", arrowsize=2]
            x -> input;
            h_in -> input;

            out_prod -> h_out

            rankdir=LR

    }"""
}

def draw_graph(name):
    graph = graphs.get(name, name)
    return graphviz.Source(graph)
