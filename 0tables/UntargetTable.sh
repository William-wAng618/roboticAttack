\begin{table*}[t]
\caption{\textbf{UnTarget Attack.} The MMAvg represents the average score on the right seven tasks. LLaVA$_{Align}$ is the stage-one LLaVA without end-to-end finetuning, and  LLaVA$_{FT}$ indicates the fully fine-tuned LLaVA. All the fine-tuned processes are using the same Vision-Flan dataset.  The best performance is in \textbf{bold}.}
\label{table:main-results}
\vspace{-1mm}
\begin{adjustbox}{width=0.91\width,center}
%\renewcommand\arraystretch{1.15}
\begin{tabular}{c|c|c|ccc}
% \renewcommand\arraystretch{1.15}
\toprule
\rowcolor{mygray}
                          &                                  &                          & \multicolumn{3}{c}{Action Target (s)}  \\
\rowcolor{mygray}
\multirow{-2}{*}{Dataset} & \multirow{-2}{*}{\# patch size}  & \multirow{-2}{*}{Metric} & DoF1 & DoF7 & DoF1-3 \\
\midrule
\midrule
\multirow{8}{*}{Bridge v2~\cite{bridgev2}} & \multirow{4}{*}{50x50} & CE                    & x & x & x  \\
                                           &                        & ASR                   & x & x & x  \\
                                           &                        & Action L1             & x & x & x  \\
                                           &                        & \cellcolor{mygray} RD & \cellcolor{mygray} x & \cellcolor{mygray} x & \cellcolor{mygray} x\\
                                           \cmidrule{2-6}
                                           & \multirow{4}{*}{22x22} & CE                    & x & x & x  \\
                                           &                        & ASR                   & x & x & x  \\
                                           &                        & Action L1             & x & x & x  \\
                                           &                        & \cellcolor{mygray} RD & \cellcolor{mygray} x & \cellcolor{mygray} x & \cellcolor{mygray} x\\
\midrule
\multirow{8}{*}{LIBERO~\cite{LIBREO}}      & \multirow{4}{*}{50x50} & CE                    & x & x & x  \\
                                           &                        & ASR                   & x & x & x  \\
                                           &                        & Action L1             & x & x & x  \\
                                           &                        & \cellcolor{mygray} RD & \cellcolor{mygray} x & \cellcolor{mygray} x & \cellcolor{mygray} x\\
                                           \cmidrule{2-6}
                                           & \multirow{4}{*}{22x22} & CE                    & x & x & x  \\
                                           &                        & ASR                   & x & x & x  \\
                                           &                        & Action L1             & x & x & x  \\
                                           &                        & \cellcolor{mygray} RD & \cellcolor{mygray} x & \cellcolor{mygray} x & \cellcolor{mygray} x\\
\bottomrule
\end{tabular}
\end{adjustbox}
\vspace{-3mm}
\end{table*}